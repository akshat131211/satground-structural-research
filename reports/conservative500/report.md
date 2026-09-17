# Smaller-learning-rate comparison

Fixed 500 steps, seed 17, learning rate 0.00003. Pending human review; no server decision.

| Model | Boundary error | IoU | LPIPS |
|---|---:|---:|---:|
| A1 | 0.195721 | 0.452819 | 0.582288 |
| A2 | 0.194261 | 0.539867 | 0.567256 |
| A3 | 0.191681 | 0.540232 | 0.567910 |

[Primary comparison](primary/report.md). [Ablations and stratum diagnostics](summary.json).
Interpret against frozen A0 and inspected reviewed views before any improvement claim.
