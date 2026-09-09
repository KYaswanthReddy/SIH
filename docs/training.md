# SIH26012 Training Guide

## 1. Overview
The SIH26012 training pipeline is designed for fast, memory-safe, and reproducible execution on local workstations and **Google Colab Pro** GPU instances (NVIDIA T4, V100, A100).

---

## 2. Dataset Geographic Partitioning
To prevent spatial data leakage, data is strictly partitioned across separate geographic aerial tiles:

| Split | Number of Tiles | Tile IDs | Prepared Patches |
|---|---|---|---|
| **Train** | 5 | `tile-100000-470000`, `tile-100000-500000`, `tile-220000-450000`, `tile-240000-460000`, `tile-180000-360000` | 2,500 |
| **Validation** | 2 | `tile-190000-350000`, `tile-160000-490000` | 600 |
| **Test** | 2 | `tile-190000-310000`, `tile-170000-460000` | 600 |

---

## 3. Loss Functions
The models use a balanced multi-objective loss:

$$L_{total} = w_{focal} L_{focal} + w_{dice} L_{dice} + w_{conn} L_{conn}$$

- **Focal Loss**: $\alpha = 0.25, \gamma = 2.0$ — mitigates extreme foreground/background class imbalance (boundary pixels $< 3\%$ of patch area).
- **Dice Loss**: Smooth factor $= 1.0$ — optimizes regional boundary overlap directly.
- **Connectivity Loss**: Masked BCE over the 8-neighborhood affinity directions.

---

## 4. Colab Pro Execution
1. Open Google Colab and set Runtime to **GPU** (T4 / A100).
2. Upload the desired notebook from `notebooks/`:
   - `notebooks/train_sanity_full_colab.ipynb` (**Phase 4 Ultra-Fast Sanity Verification**: 50 train, 10 val, 10 test @ 256×256, 1 epoch, <1 min on GPU)
   - `notebooks/train_baseline_colab.ipynb` (Phase 2 Baseline)
   - `notebooks/train_pmg_colab.ipynb` (Phase 3 PMG)
   - `notebooks/train_full_colab.ipynb` (Phase 5 Full Model Production)
3. Run all cells sequentially.
4. Checkpoints will be automatically saved to `experiments/checkpoints/best_model.pth`.

---

## 5. Local Training & Sanity Runs
```bash
# Run ultra-fast full architecture sanity check (50 train, 10 val, 10 test, 256x256, 1 epoch)
.venv/bin/python scripts/train_full.py configs/sanity_full.yaml

# Run 2-epoch baseline sanity verification
.venv/bin/python scripts/train_sanity.py

# Run full baseline training (15 epochs)
.venv/bin/python -m ai.trainer --config configs/baseline_train_config.yaml

# Run full architecture training (15 epochs)
.venv/bin/python scripts/train_full.py configs/full.yaml
```
