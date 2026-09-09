"""
Backend Inference Service for Cadastral Boundary Models (Baseline, PMG, DG, Full).

Preprocessing is kept strictly identical to the offline evaluation pipeline
(ai/transforms.py CadastreAugmentation val path):
  - Divide by 255.0 → float32 [0, 1]
  - cv2.INTER_LINEAR resize to model's native training resolution (256×256)
  - NO ImageNet mean/std normalization (training pipeline does not apply it)
  - np.transpose (H, W, C) → (C, H, W), torch.from_numpy, unsqueeze(0)
  - Output head: refined_prob (connectivity-fused boundary prediction)
  - Bilinear upsample back to original (512×512) GIS coordinate space
"""

import os
import cv2
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from ai.models.unet_baseline import build_baseline_model, CadastreUNetBaseline
from ai.models.pmg import CadastreUNetPMG
from ai.models.domain_generalization import CadastreUNetDG
from ai.models.full_model import CadastreUNetFull


# ---------------------------------------------------------------------------
# Shared preprocessing function — identical to ai/transforms.py val path
# ---------------------------------------------------------------------------

def preprocess_for_inference(
    image_np: np.ndarray,
    target_hw: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    Preprocess a uint8 RGB image for model inference.

    This function is the single, authoritative source of preprocessing logic.
    It is used by both the offline evaluation pipeline (ai/evaluate.py via
    ai/transforms.py CadastreAugmentation with is_train=False) and the WebGIS
    InferenceService.predict(), ensuring numerical identity between both paths.

    Pipeline (exactly matching ai/transforms.py val path):
      1. Cast to float32, divide by 255.0 → [0.0, 1.0]
      2. cv2.resize to target_hw with INTER_LINEAR interpolation
      3. np.transpose (H, W, C) → (C, H, W)
      4. torch.from_numpy → float32 tensor, unsqueeze(0) → (1, C, H, W)

    No ImageNet mean/std normalization is applied; the training pipeline does
    not use it (CadastreAugmentation does not subtract mean or divide by std).

    Args:
        image_np: (H, W, 3) uint8 RGB numpy array
        target_hw: (height, width) tuple for resize target

    Returns:
        Tensor of shape (1, 3, target_h, target_w), dtype=float32, in [0, 1]
    """
    th, tw = target_hw
    img_f32 = image_np.astype(np.float32) / 255.0

    h, w = img_f32.shape[:2]
    if (h, w) != (th, tw):
        img_f32 = cv2.resize(img_f32, (tw, th), interpolation=cv2.INTER_LINEAR)

    img_chw = np.transpose(img_f32, (2, 0, 1))          # (3, H, W)
    tensor  = torch.from_numpy(img_chw).unsqueeze(0)    # (1, 3, H, W)
    return tensor.float()


class InferenceService:
    """
    Manages model loading, caching, and inference across all model variants.

    The predict() method uses preprocess_for_inference() — the same function
    that the offline evaluation pipeline uses — to guarantee numerical parity.
    """

    def __init__(self, config_path: str = "configs/baseline_train_config.yaml"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        self.models: Dict[str, torch.nn.Module] = {}
        self._native_size: Tuple[int, int] = (256, 256)  # quick_full_fixed training resolution
        self.load_models()

    def load_models(self):
        """Initialize all model variants."""
        # 1. Baseline
        baseline = CadastreUNetBaseline(in_channels=3, out_channels=1, features=[32, 64, 128, 256, 512])
        ckpt_path = "experiments/sanity/checkpoints/best_model.pth"
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                baseline.load_state_dict(state_dict)
            except Exception as e:
                print(f"[InferenceService] Note: Using initialized baseline model ({e})")
        baseline.to(self.device).eval()
        self.models["baseline"] = baseline

        # 2. PMG Model
        pmg_model = CadastreUNetPMG(in_channels=3, out_channels=1, features=[32, 64, 128, 256, 512], pmg_enabled=True)
        pmg_model.to(self.device).eval()
        self.models["pmg"] = pmg_model

        # 3. PMG + DG Model
        dg_model = CadastreUNetDG(in_channels=3, out_channels=1, features=[32, 64, 128, 256, 512], pmg_enabled=True, dg_enabled=True)
        dg_model.to(self.device).eval()
        self.models["pmg_dg"] = dg_model

        # 4. Full Model (PMG + DG + Connectivity) — trained checkpoint required
        full_model = CadastreUNetFull(
            in_channels=3,
            out_channels=1,
            features=[32, 64, 128, 256, 512],
            pmg_enabled=True,
            dg_enabled=True,
            connectivity_enabled=True,
            dg_prob=0.0,    # Disable DG at inference time (eval mode handles this)
        )
        # Check for 512x512 full production checkpoint first, then fallback to 256x256 quick run
        ckpt_512_path = "experiments/full_512_fixed/results/checkpoints/best_model.pth"
        ckpt_quick_path = "experiments/quick_full_fixed/results/checkpoints/best_model.pth"
        if os.path.exists(ckpt_512_path):
            ckpt_full_path = ckpt_512_path
            self._native_size = (512, 512)
        elif os.path.exists(ckpt_quick_path):
            ckpt_full_path = ckpt_quick_path
            self._native_size = (256, 256)
        else:
            raise FileNotFoundError(
                f"[InferenceService] ERROR: Required trained checkpoint not found at: {ckpt_512_path} or {ckpt_quick_path}. "
                f"Cannot start WebGIS backend with untrained random weights."
            )

        ckpt_full = torch.load(ckpt_full_path, map_location=self.device)
        state_dict = ckpt_full.get("model_state_dict", ckpt_full)
        full_model.load_state_dict(state_dict)
        full_model.to(self.device).eval()
        self.models["full"] = full_model

        # Extract checkpoint metadata
        epoch  = ckpt_full.get("epoch", "N/A")
        val_f1 = ckpt_full.get("val_metrics", {}).get("f1", "N/A")
        val_th = ckpt_full.get("selected_threshold",
                               ckpt_full.get("val_metrics", {}).get("selected_threshold", 0.50))

        self.checkpoint_info = {
            "path": ckpt_full_path,
            "epoch": epoch,
            "val_f1": val_f1,
            "selected_threshold": val_th,
            "device": str(self.device),
        }

        print("=" * 60)
        print(" [InferenceService] ⭐ TRAINED FULL MODEL LOADED SUCCESSFULLY")
        print(f" • Checkpoint Path:               {ckpt_full_path}")
        print(f" • Checkpoint Epoch:              {epoch}")
        print(f" • Best Validation F1:            {val_f1}")
        print(f" • Selected Validation Threshold: {val_th}")
        print(f" • Native Training Resolution:    {self._native_size[0]}×{self._native_size[1]}")
        print(f" • Compute Device:                {self.device}")
        print(" • Preprocessing:                 /255 → INTER_LINEAR resize → CHW (no ImageNet norm)")
        print("=" * 60)
        print(f"[InferenceService] Loaded {len(self.models)} model architectures on device: {self.device}")

    def predict(
        self,
        image_np: np.ndarray,
        model_name: str = "baseline",
    ) -> Dict[str, Any]:
        """
        Run inference on an RGB image.

        Preprocessing is performed by preprocess_for_inference(), which is
        the single authoritative implementation shared with the offline
        evaluation pipeline (ai/transforms.py val path). This guarantees that
        WebGIS and offline evaluation produce numerically identical probability
        maps for the same image.

        Steps:
          1. preprocess_for_inference() → (1, 3, 256, 256) tensor in [0, 1]
          2. model forward pass → refined_prob at native 256×256 resolution
          3. Bilinear upsample to original (512×512) GIS coordinate space
          4. Connectivity tensor similarly upsampled for full model

        Args:
            image_np: (H, W, 3) uint8 RGB image
            model_name: one of 'baseline', 'pmg', 'pmg_dg', 'full'

        Returns:
            Dict with:
              'prob_map'          : (orig_H, orig_W) float32 probability map in [0, 1]
              'prob_map_native'   : (256, 256) float32 — raw model output at training resolution
              'connectivity'      : (8, orig_H, orig_W) float32 or None
              'mean_confidence'   : scalar float
              'max_confidence'    : scalar float
              'model_name'        : str
        """
        model = self.models.get(model_name, self.models["baseline"])
        orig_h, orig_w = image_np.shape[:2]

        # ── Step 1: Preprocess (identical to offline pipeline) ─────────────
        tensor = preprocess_for_inference(image_np, target_hw=self._native_size).to(self.device)

        # ── Step 2: Forward pass ───────────────────────────────────────────
        with torch.no_grad():
            if model_name == "full" and isinstance(model, CadastreUNetFull):
                out_dict = model(tensor, return_dict=True)
                prob_native = out_dict["refined_prob"].squeeze().cpu().numpy()
                conn_native = (out_dict["connectivity_prob"].squeeze().cpu().numpy()
                               if out_dict["connectivity_prob"] is not None else None)
            else:
                prob_native = model(tensor).squeeze().cpu().numpy()
                conn_native = None

        # ── Step 3: Upsample to GIS coordinate space (512×512) ────────────
        if (orig_h, orig_w) != self._native_size:
            prob_map = cv2.resize(prob_native, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if conn_native is not None:
                conn_map = np.zeros((conn_native.shape[0], orig_h, orig_w), dtype=np.float32)
                for d in range(conn_native.shape[0]):
                    conn_map[d] = cv2.resize(conn_native[d], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                conn_map = None
        else:
            prob_map  = prob_native
            conn_map  = conn_native

        return {
            "prob_map":        prob_map,         # (orig_H, orig_W) for GIS vectorization
            "prob_map_native": prob_native,       # (256, 256) raw model output
            "connectivity":    conn_map,
            "mean_confidence": float(prob_map.mean()),
            "max_confidence":  float(prob_map.max()),
            "model_name":      model_name,
        }
