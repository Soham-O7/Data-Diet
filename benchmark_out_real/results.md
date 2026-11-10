
### mnist

Full training set: 16000 rows, test accuracy 0.9775, fit 11.28 CPU-s.

| keep | method | accuracy retained (mean ± sd) | test acc | prune CPU-s | est. prune kWh | break-even runs | selector(s) |
|---|---|---|---|---|---|---|---|
| 20% | random | 98.8% ± 0.2% | 0.9657 | 145.3 | 6.06e-04 | 13.6 | random |
| 20% | el2n | 99.8% ± 0.0% | 0.9760 | 161.7 | 6.94e-04 | 16.8 | el2n |
| 20% | ccs | 97.7% ± 0.2% | 0.9554 | 161.9 | 6.85e-04 | 15.0 | ccs |
| 20% | d2 | 99.6% ± 0.0% | 0.9739 | 163.7 | 6.96e-04 | 15.9 | d2 |
| 20% | clean_random | 98.9% ± 0.2% | 0.9666 | 175.3 | 7.37e-04 | 16.1 | random |
| 20% | clean_el2n | **100.2% ± 0.0%** | 0.9793 | 188.8 | 8.04e-04 | 18.6 | el2n |
| 20% | full_mix | 99.8% ± 0.0% | 0.9758 | 194.4 | 8.27e-04 | 19.2 | d2 |
| 40% | random | 99.3% ± 0.2% | 0.9711 | 145.3 | 6.06e-04 | 14.6 | random |
| 40% | el2n | **100.1% ± 0.0%** | 0.9788 | 161.0 | 6.83e-04 | 18.5 | el2n |
| 40% | ccs | 98.0% ± 0.1% | 0.9582 | 161.9 | 6.85e-04 | 15.4 | ccs |
| 40% | d2 | 99.9% ± 0.1% | 0.9768 | 163.2 | 6.89e-04 | 17.1 | d2 |
| 40% | clean_random | 99.4% ± 0.0% | 0.9718 | 173.6 | 7.25e-04 | 16.7 | random |
| 40% | clean_el2n | 100.1% ± 0.1% | 0.9782 | 187.5 | 7.86e-04 | 19.4 | el2n |
| 40% | full_mix | 100.1% ± 0.1% | 0.9782 | 192.3 | 8.07e-04 | 21.2 | el2n |
| 60% | random | 99.7% ± 0.1% | 0.9749 | 145.3 | 6.06e-04 | 19.2 | random |
| 60% | el2n | 100.1% ± 0.0% | 0.9782 | 162.0 | 6.80e-04 | 22.5 | el2n |
| 60% | ccs | 98.3% ± 0.2% | 0.9612 | 162.3 | 6.80e-04 | 16.7 | ccs |
| 60% | d2 | **100.2% ± 0.0%** | 0.9791 | 164.3 | 6.85e-04 | 23.4 | d2 |
| 60% | clean_random | 99.7% ± 0.1% | 0.9743 | 173.7 | 7.22e-04 | 20.1 | random |
| 60% | clean_el2n | 99.9% ± 0.1% | 0.9768 | 187.8 | 7.86e-04 | 22.7 | el2n |
| 60% | full_mix | 99.9% ± 0.1% | 0.9768 | 195.2 | 8.11e-04 | 24.1 | el2n |

**bold** = best mean retained accuracy at that keep ratio. With few seeds and small datasets, gaps of about one standard deviation are not evidence of a real difference.
