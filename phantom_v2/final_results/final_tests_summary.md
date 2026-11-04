# Test finali (seed 314159)

## Test held-out (2000 campioni, ±0.17 rad, ±10 mm, σ = 0.3 mm)

| Configurazione | Rot. [°] | Trasl. [mm] | TRE media [mm] | TRE p95 [mm] | Success (TRE < 5 mm) |
|---|---|---|---|---|---|
| Sweep r=60, solo pose | 5.77 | 7.45 | 8.06 | 15.38 | 22.7% |
| Sweep r=60, pose + KL 0.03 | 4.21 | 5.99 | 6.00 | 11.98 | 44.1% |
| Sparse n=50, solo pose | 2.68 | 3.83 | 3.69 | 6.74 | 80.1% |
| Sparse n=50, pose + KL 0.10 | 1.72 | 2.11 | 1.80 | 3.55 | 98.8% |
| Baseline 4 landmark | 0.43 | 0.56 | 0.40 | 0.77 | 100.0% |

## Bacino di convergenza (1000 campioni per livello)

| Configurazione | Disallineamento max per asse | TRE media [mm] | TRE p95 [mm] | Success DCP | Success baseline |
|---|---|---|---|---|---|
| Sweep r=60, pose + KL 0.03 | ±10°, ±10 mm | 5.94 | 11.35 | 44.2% | 100.0% |
| Sweep r=60, pose + KL 0.03 | ±20°, ±20 mm | 14.49 | 31.37 | 7.8% | 100.0% |
| Sweep r=60, pose + KL 0.03 | ±30°, ±30 mm | 27.08 | 60.79 | 1.4% | 100.0% |
| Sweep r=60, pose + KL 0.03 | ±45°, ±50 mm | 49.36 | 104.58 | 0.1% | 100.0% |
| Sparse n=50, pose + KL 0.10 | ±10°, ±10 mm | 1.82 | 3.58 | 99.2% | 100.0% |
| Sparse n=50, pose + KL 0.10 | ±20°, ±20 mm | 4.76 | 12.42 | 67.4% | 100.0% |
| Sparse n=50, pose + KL 0.10 | ±30°, ±30 mm | 12.93 | 32.33 | 21.7% | 100.0% |
| Sparse n=50, pose + KL 0.10 | ±45°, ±50 mm | 28.69 | 62.50 | 1.9% | 100.0% |

## Robustezza al rumore (σ per coordinata)

| Configurazione | σ [mm] | TRE media DCP [mm] | Success DCP | TRE media baseline [mm] |
|---|---|---|---|---|
| Sweep r=60, pose + KL 0.03 | 0.0 | 5.91 | 45.0% | 0.00 |
| Sweep r=60, pose + KL 0.03 | 0.3 | 6.00 | 44.1% | 0.40 |
| Sweep r=60, pose + KL 0.03 | 0.6 | 5.98 | 43.5% | 0.80 |
| Sparse n=50, pose + KL 0.10 | 0.0 | 1.81 | 99.2% | 0.00 |
| Sparse n=50, pose + KL 0.10 | 0.3 | 1.80 | 98.8% | 0.40 |
| Sparse n=50, pose + KL 0.10 | 0.6 | 1.85 | 99.2% | 0.80 |
