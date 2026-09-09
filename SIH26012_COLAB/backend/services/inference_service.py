"""
Backend Inference Service for Cadastral Boundary Models (Baseline, PMG, DG, Full).
"""

import os
import yaml
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from ai.models.unet_baseline import build_baseline_model, CadastreUNetBaseline
from ai.models.pmg import CadastreUNetPMG
from ai.models.domain_generalization import CadastreUNetDG
from ai.models.full_model import CadastreUNetFull


class InferenceService:
    """
    Manages model loading, caching, and inference across all model variants.
    """

    def __init__(self, config_path: str = "configs/baseline_train_config.yaml"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        self.models: Dict[str, torch.nn.Module] = {}
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

        # 4. Full Model (PMG + DG + Connectivity)
        full_model = CadastreUNetFull(
            in_channels=3,
            out_channels=1,
            features=[32, 64, 128, 256, 512],
            pmg_enabled=True,
            dg_enabled=True,
            connectivity_enabled=True,
            dg_prob=0.0
        )
        ckpt_full_path = "experiments/quick_full_fixed/results/checkpoints/best_model.pth"
        if not os.path.exists(ckpt_full_path):
            raise FileNotFoundError(
                f"[InferenceService] ERROR: Required trained checkpoint not found at: {ckpt_full_path}. "
                f"Cannot start WebGIS backend with untrained random weights."
            )
        
        ckpt_full = torch.load(ckpt_full_path, map_location=self.device)
        state_dict = ckpt_full.get("model_state_dict", ckpt_full)
        full_model.load_state_dict(state_dict)
        full_model.to(self.device).eval()
        self.models["full"] = full_model

        # Extract checkpoint metadata
        epoch = ckpt_full.get("epoch", "N/A")
        val_f1 = ckpt_full.get("val_metrics", {}).get("f1", "N/A")
        val_th = ckpt_full.get("selected_threshold", ckpt_full.get("val_metrics", {}).get("selected_threshold", 0.50))
        
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
        print(f" • Compute Device:                {self.device}")
        print("=" * 60)
        print(f"[InferenceService] Loaded {len(self.models)} model architectures on device: {self.device}")

    def predict(
        self,
        image_np: np.ndarray,
        model_name: str = "baseline"
    ) -> Dict[str, Any]:
        """
        Run inference on RGB image, aligning input resolution to model's native training size (256x256)
        and restoring full-resolution probability map (512x512) for GIS vectorization.
        
        Args:
            image_np: (H, W, 3) uint8 RGB image
            model_name: 'baseline', 'pmg', 'pmg_dg', or 'full'
            
        Returns:
            Dict with 'prob_map' (HxW float32 in [0, 1]), 'connectivity' (if full model)
        """
        import cv2
        model = self.models.get(model_name, self.models["baseline"])
        orig_h, orig_w = image_np.shape[:2]
        target_size = (256, 256) # Native resolution for quick_full_fixed checkpoint

        # 1. Preprocess: Resize to model's native training receptive field
        if (orig_h, orig_w) != target_size:
            img_resized = cv2.resize(image_np.astype(np.float32) / 255.0, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = image_np.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).unsqueeze(0).to(self.device).float()

        # 2. Forward pass
        with torch.no_grad():
            if model_name == "full" and isinstance(model, CadastreUNetFull):
                out_dict = model(img_tensor, return_dict=True)
                prob_raw = out_dict["refined_prob"].squeeze().cpu().numpy()
                conn = out_dict["connectivity_prob"].squeeze().cpu().numpy() if out_dict["connectivity_prob"] is not None else None
            else:
                prob_raw = model(img_tensor).squeeze().cpu().numpy()
                conn = None

        # 3. Postprocess: Upsample probability map to original 512x512 GIS coordinates
        if (orig_h, orig_w) != target_size:
            prob = cv2.resize(prob_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if conn is not None:
                conn_resized = np.zeros((conn.shape[0], orig_h, orig_w), dtype=np.float32)
                for d in range(conn.shape[0]):
                    conn_resized[d] = cv2.resize(conn[d], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                conn = conn_resized
        else:
            prob = prob_raw

        return {
            "prob_map": prob,
            "connectivity": conn,
            "mean_confidence": float(prob.mean()),
            "max_confidence": float(prob.max()),
            "model_name": model_name
        }

