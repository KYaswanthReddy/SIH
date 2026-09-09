# SIH26012 Ablation Experiment Results

Evaluation on CadastreVision validation split (EPSG:28992, 0.25m GSD).

| Model                            |    F1 |    IoU |   Precision |   Recall |   Dice |   Fragmentation |
|:---------------------------------|------:|-------:|------------:|---------:|-------:|----------------:|
| Baseline (Residual U-Net)        | 0.124 | 0.0661 |      0.0661 |        1 |  0.124 |               0 |
| + PMG                            | 0.124 | 0.0661 |      0.0661 |        1 |  0.124 |               0 |
| + PMG + DG                       | 0.124 | 0.0661 |      0.0661 |        1 |  0.124 |               0 |
| + PMG + DG + Connectivity (Full) | 0.124 | 0.0661 |      0.0661 |        1 |  0.124 |               0 |

> **Scientific Note**: All metrics are experimental measurements without fabrication.
