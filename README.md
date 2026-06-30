# Cell-Free Massive MIMO Systems
### EEN1095 —  Electronic Engineering Project

**Student:** Rahul Putta
**Student ID:** A00057
**Supervisor:** Sobia Jangsher
**Period:** May – August 2026

---

## Project Overview

This project investigates **Cell-Free Massive MIMO (CF-mMIMO)** systems and evaluates their performance against conventional small-cell networks. Unlike traditional cellular architectures, CF-mMIMO deploys a large number of geographically distributed Access Points (APs) that serve all User Equipment (UEs) simultaneously over the same time-frequency resources, coordinated by a Central Processing Unit (CPU).

## Objective:
To demonstrate that CF-mMIMO delivers uniformly great service for all users — particularly improving 95th-percentile (edge-user) rates compared to small-cell baselines.


## Current Progress (Weeks 1–8: ~30%)

### Completed
| Module | Description | Status |
|--------|-------------|--------|
| Channel model | Large-scale fading (path-loss + log-normal shadowing) | Done |
| Uplink SE | Closed-form MF combining SE (Ngo 2017, Theorem 1) | Done |
| Power control | Max-min fairness via bisection + heuristic allocation | Done |
| Literature review | 11 key CF-mMIMO papers reviewed | Done |

### Remaining
| Module | Description | Target |
|--------|-------------|--------|
| Downlink MF | Conjugate beamforming SE | Week 9-10 |
| Downlink ZF | Zero-forcing precoding + SOCP power opt. | Week 10-11 |
| Small-cell baseline | Independent cells, nearest-AP association | Week 11 |
| Monte Carlo loop | 200+ drop averaging, full CDF generation | Week 11-12 |
| IEEE paper | Conference-style paper (5-10 pages) | Week 13-14 |

---

## Key Results So Far

Single-drop uplink SE (L=64 APs, K=16 UEs, 1000×1000 m²):

| Scheme | Mean SE (bit/s/Hz) | Min SE (bit/s/Hz) | 5th-pctile SE |
|--------|---------------------|---------------------|-----------------|
| Equal power | 2.501 | 0.006 | 0.014 |
| Heuristic/max-min power control | 2.776 | 1.734 | 1.809 |

The min-SE improvement (0.006 → 1.734 bit/s/Hz) demonstrates the max-min power control benefit, consistent with Ngo et al. (2017).

---

## How to Run

```bash
pip install numpy matplotlib
python src/cf_mmimo_simulation.py
```

Output: prints SE statistics table and saves `results/uplink_SE_CDF.png`.

---

## Key References

1. H. Q. Ngo et al., "Cell-Free Massive MIMO Systems," IEEE Trans. Wireless Commun., 2017.
2. E. Nayebi et al., "Precoding and power optimization in cell-free massive MIMO systems," IEEE Trans. Wireless Commun., 2017.
3. Ö. T. Demir, E. Björnson, L. Sanguinetti, "Foundations of User-Centric Cell-Free Massive MIMO," Found. Trends Signal Process., 2021.
4. Z. Chen, E. Björnson, "Cell-Free Massive MIMO: A Survey," IEEE Commun. Surveys Tuts., 2022.

Full reference list (11 papers) in `docs/literature_notes.md`.

---


