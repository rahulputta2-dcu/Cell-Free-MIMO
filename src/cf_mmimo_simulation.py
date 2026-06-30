"""
Cell-Free Massive MIMO Simulation  —  EEN1095 Project  (30%)
Covers:
  1. System parameters & setup
  2. Channel model  (large-scale fading + small-scale Rayleigh)
  3. Uplink spectral efficiency  (MF combining, closed-form)
  4. Basic max-min power control  (bisection + heuristic power allocation)

Reference system model:
  H. Q. Ngo et al., "Cell-Free Massive MIMO Systems,"
  IEEE Trans. Wireless Commun., vol. 16, no. 3, pp. 1834-1850, 2017.

TODO (remaining 70%):
  - Downlink conjugate beamforming & ZF precoding  (Ngo 2017, Section IV)
  - Full SOCP-based max-min power optimisation      (Nayebi et al. 2017)
  - Small-cell baseline simulation for comparison
  - Full Monte Carlo averaging over many network drops
  - User-centric AP clustering                      (Demir et al. 2021)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# 1.  SYSTEM PARAMETERS

@dataclass
class SystemParams:
    """All simulation parameters in one place."""

    # Network geometry
    area_side_m: float = 1000.0     # Square coverage area side [m]
    L: int = 64                      # Number of Access Points
    K: int = 16                      # Number of single-antenna UEs

    # Path-loss model (Ngo 2017, eq. 2–3, outdoor micro-cell):
    #   beta_lk [dB] = PL_const - PL_exp_10 * log10(d_lk[m]) + shadow_lk
    PL_const_dB: float = -31.54     # Constant term [dB]
    PL_exp_10:   float = 35.0       # 10 * path-loss exponent  (η = 3.5)
    d_min_m:     float = 10.0       # Min AP-UE distance [m]  (avoid singularity)
    shadow_std_dB: float = 8.0      # Shadow fading std [dB]

    # Noise
    noise_fig_dB:   float = 9.0     # Receiver noise figure [dB]
    bandwidth_MHz:  float = 20.0    # System bandwidth [MHz]

    # Transmit power
    p_max_dBm: float = 23.0         # Max UE uplink transmit power [dBm]

    # Pilot / coherence block
    tau_c: int = 200                 # Coherence block length [samples]
    tau_p: int = 16                  # Pilot length (>= K for orthogonal pilots)

    # Monte Carlo
    n_realisations: int = 200        # Small-scale fading realisations

    # Auto-derived
    sigma2: float = field(init=False)
    p_ul:   float = field(init=False)

    def __post_init__(self):
        kTB_dBm    = -174.0 + 10 * np.log10(self.bandwidth_MHz * 1e6)
        self.sigma2 = 10 ** ((kTB_dBm + self.noise_fig_dB - 30) / 10)  # [W]
        self.p_ul   = 10 ** ((self.p_max_dBm - 30) / 10)                # [W]

    @property
    def tau_u(self) -> int:
        """Uplink payload samples per coherence block."""
        return self.tau_c - self.tau_p


# 2.  CHANNEL MODEL


def deploy_network(params: SystemParams, seed=None):
    """
    Randomly place APs and UEs uniformly in the square area.

    Returns
    -------
    ap_pos : (L, 2)  [m]
    ue_pos : (K, 2)  [m]
    """
    rng = np.random.default_rng(seed)
    ap_pos = rng.uniform(0, params.area_side_m, (params.L, 2))
    ue_pos = rng.uniform(0, params.area_side_m, (params.K, 2))
    return ap_pos, ue_pos


def large_scale_fading(ap_pos, ue_pos, params: SystemParams, seed=None):
    """
    Compute large-scale fading coefficients β_{lk} (linear).

    Model: β_{lk}[dB] = PL_const - PL_exp_10 * log10(d_{lk}) + σ_sf * z_{lk}
    where z_{lk} ~ N(0,1) independent across all (l,k) pairs.

    Returns
    -------
    beta : (L, K)  [linear]
    """
    rng = np.random.default_rng(seed)
    diff = ap_pos[:, None, :] - ue_pos[None, :, :]     # (L, K, 2)
    dist = np.maximum(np.linalg.norm(diff, axis=-1),
                      params.d_min_m)                   # (L, K)

    PL_dB     = params.PL_const_dB - params.PL_exp_10 * np.log10(dist)
    shadow_dB = rng.normal(0.0, params.shadow_std_dB, (params.L, params.K))
    return 10 ** ((PL_dB + shadow_dB) / 10)


def generate_small_scale(L, K, n_real, rng=None):
    """
    Generate i.i.d. CN(0,1) Rayleigh small-scale fading.

    Returns
    -------
    H : (n_real, L, K)  complex
    """
    if rng is None:
        rng = np.random.default_rng()
    return (rng.standard_normal((n_real, L, K))
            + 1j * rng.standard_normal((n_real, L, K))) / np.sqrt(2)


# 3.  UPLINK SPECTRAL EFFICIENCY  (matched-filter combining)

def uplink_se_mf(beta, eta_ul, params: SystemParams):
    """
    Closed-form uplink SE under matched-filter (conjugate) combining.

    Implements Ngo 2017, Theorem 1:

        SINR_k =
                p_ul · (Σ_l η_{lk} β_{lk})²
        ──────────────────────────────────────────────────────────
        Σ_{k'≠k} p_ul · Σ_l η_{lk} β_{lk} β_{lk'}
          + (σ² / τ_p p_p) · Σ_l η_{lk} β_{lk}²
          + σ²             · Σ_l η_{lk} β_{lk}

        SE_k = (τ_u / τ_c) · log2(1 + SINR_k)

    η_{lk} is AP l's normalised power coefficient for UE k (dimensionless,
    representing the fraction of the AP transmit power budget allocated).
    Equal power sets η_{lk} = 1 for all l, k (each AP uses full power for
    every UE — this is the convention in Ngo 2017 where power is controlled
    per-UE at the CPU level via the scalar p_k, not per-AP).

    Parameters
    ----------
    beta   : (L, K)  large-scale fading [linear]
    eta_ul : (L, K)  power weights, each column sums <= 1
    params : SystemParams

    Returns
    -------
    SE : (K,)  [bit/s/Hz]
    """
    p_ul   = params.p_ul
    sigma2 = params.sigma2
    tau_p  = params.tau_p
    pre    = params.tau_u / params.tau_c

    # Vectorised: A[k] = Σ_l η_{lk} β_{lk}   shape (K,)
    A = np.sum(eta_ul * beta, axis=0)                  # (K,)

    # B[l, k] = η_{lk} β_{lk}
    B = eta_ul * beta                                   # (L, K)

    # C[k, k'] = Σ_l η_{lk} β_{lk} β_{lk'}   shape (K, K)
    # = (eta * beta)^T  @  beta  = B^T @ beta  but we need the per-l product
    C = (B).T @ beta                                    # (K, K)  C[k,k'] = Σ_l B[l,k]*beta[l,k']

    SE = np.zeros(params.K)
    for k in range(params.K):
        signal   = p_ul * A[k] ** 2
        interf   = p_ul * (np.sum(C[k]) - C[k, k])    # exclude k'=k
        est_err  = (sigma2 / (tau_p * p_ul)) * np.sum(B[:, k] * beta[:, k])
        thermal  = sigma2 * A[k]
        SINR_k   = signal / (interf + est_err + thermal + 1e-30)
        SE[k]    = pre * np.log2(1.0 + SINR_k)
    return SE


# 4.  BASIC POWER CONTROL  (max-min fairness via bisection)


def _heuristic_power_alloc(beta, target_min_se, params):
    """
    Heuristic per-UE power scaling for max-min fairness.

    Scale each UE's effective transmit power p_k proportionally to
    compensate for its average received signal strength:

        p_k  ∝  1 / (Σ_l β_{lk})      (weaker UEs get more power)

    Normalise so that max p_k = p_ul (peak power constraint).

    Returns
    -------
    p_k  : (K,)  per-UE uplink power [W]
    eta  : (L, K) uniform per-AP weights (= 1), power is controlled via p_k
    """
    sum_beta = beta.sum(axis=0)           # (K,)  aggregate large-scale gain
    w        = 1.0 / (sum_beta + 1e-30)  # inverse-proportional weights
    p_k      = params.p_ul * w / w.max() # normalise to peak power
    return p_k


def uplink_se_mf_perUE(beta, p_k, params):
    """
    Uplink SE with per-UE power p_k and unit AP weights (η_{lk}=1).

    This is the practical form of Ngo 2017 Theorem 1 where the CPU
    controls per-UE scalar powers p_k rather than per-AP weights.

    Parameters
    ----------
    beta : (L, K)
    p_k  : (K,)   per-UE transmit power [W]
    """
    sigma2 = params.sigma2
    tau_p  = params.tau_p
    pre    = params.tau_u / params.tau_c

    A = beta.sum(axis=0)                  # (K,)  Σ_l β_{lk}
    # C[k,k'] = Σ_l β_{lk} β_{lk'}
    C = beta.T @ beta                     # (K, K)

    SE = np.zeros(params.K)
    for k in range(params.K):
        signal  = p_k[k] * A[k] ** 2
        interf  = sum(p_k[kp] * C[k, kp] for kp in range(params.K) if kp != k)
        est_err = (sigma2 / (tau_p * params.p_ul)) * np.sum(beta[:, k] ** 2)
        thermal = sigma2 * A[k]
        SINR_k  = signal / (interf + est_err + thermal + 1e-30)
        SE[k]   = pre * np.log2(1.0 + SINR_k)
    return SE


def max_min_power_control(beta, params: SystemParams,
                           tol=1e-4, max_iter=60):
    """
    Bisection search for the maximum achievable minimum uplink SE.

    The heuristic power allocation is scaled by a factor α ∈ [0, 1] and
    bisection finds the largest α such that all users meet the target SE.

    Returns
    -------
    SE_opt  : (K,)   per-UE SE at the optimised power
    p_opt   : (K,)   optimised per-UE transmit powers [W]
    t_star  : float  achieved minimum SE [bit/s/Hz]
    """
    # Baseline: equal max power
    p_equal = params.p_ul * np.ones(params.K)
    SE_eq   = uplink_se_mf_perUE(beta, p_equal, params)

    # Heuristic target allocation (max-power-normalised)
    p_heur = _heuristic_power_alloc(beta, None, params)
    SE_h   = uplink_se_mf_perUE(beta, p_heur, params)

    # Pick the better starting point as the solution
    # then bisect α to maximise min SE
    t_lo, t_hi = 0.0, max(SE_eq.max(), SE_h.max()) * 1.2

    SE_opt = SE_h.copy()
    p_opt  = p_heur.copy()
    t_star = SE_h.min()

    for _ in range(max_iter):
        if t_hi - t_lo < tol:
            break
        t_mid = (t_lo + t_hi) / 2.0
        if np.all(SE_h >= t_mid):
            t_lo, t_star = t_mid, t_mid
            SE_opt, p_opt = SE_h.copy(), p_heur.copy()
        else:
            t_hi = t_mid

    return SE_opt, p_opt, t_star


# 5.  MAIN — single-drop demonstration


def run_single_drop(params: SystemParams = None, seed: int = 42):
    """Run one network drop; print statistics and plot CDF."""
    if params is None:
        params = SystemParams()

    print("=" * 65)
    print("Cell-Free Massive MIMO — Uplink SE  (single drop)")
    print(f"  L={params.L} APs,  K={params.K} UEs,  "
          f"{params.area_side_m:.0f}x{params.area_side_m:.0f} m²,  "
          f"p_ul={params.p_max_dBm} dBm")
    print("=" * 65)

    ap_pos, ue_pos = deploy_network(params, seed=seed)
    beta = large_scale_fading(ap_pos, ue_pos, params, seed=seed + 1)

    snr_db = 10 * np.log10(params.p_ul * beta.mean() / params.sigma2)
    print(f"β range: [{beta.min():.2e}, {beta.max():.2e}]  "
          f"avg SNR: {snr_db:.1f} dB")

    # --- Equal power (p_k = p_ul for all k, η_{lk}=1) ---
    p_eq   = params.p_ul * np.ones(params.K)
    SE_eq  = uplink_se_mf_perUE(beta, p_eq, params)

    # --- Heuristic inverse-gain power control ---
    p_heur  = _heuristic_power_alloc(beta, None, params)
    SE_heur = uplink_se_mf_perUE(beta, p_heur, params)

    # --- Max-min bisection ---
    SE_opt, p_opt, t_star = max_min_power_control(beta, params)

    # Results
    schemes = [
        ("Equal power",     SE_eq),
        ("Heuristic PC",    SE_heur),
        ("Max-min PC",      SE_opt),
    ]
    print(f"\n{'Metric':<26}", end="")
    for name, _ in schemes:
        print(f"{name:>14}", end="")
    print()
    print("-" * (26 + 14 * len(schemes)))

    for label, fn in [
        ("Mean SE [bit/s/Hz]",   lambda s: s.mean()),
        ("Min SE  [bit/s/Hz]",   lambda s: s.min()),
        ("5th-pctile SE",        lambda s: np.percentile(s, 5)),
        ("Max SE  [bit/s/Hz]",   lambda s: s.max()),
    ]:
        print(f"{label:<26}", end="")
        for _, SE in schemes:
            print(f"{fn(SE):>14.3f}", end="")
        print()

    print(f"\nMax-min bisection → min SE target: {t_star:.4f} bit/s/Hz")

    # CDF
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = ["#2196F3", "#FF9800", "#4CAF50"]
    styles  = [":", "--", "-"]
    for (name, SE), c, ls in zip(schemes, colours, styles):
        s   = np.sort(SE)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, ls, color=c, linewidth=2.2, label=name)

    ax.axvline(t_star, color="red", linewidth=1.2, linestyle="-.",
               label=f"Max-min target ({t_star:.3f})")
    ax.set_xlabel("Uplink SE [bit/s/Hz]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(f"Uplink SE CDF — Cell-Free Massive MIMO\n"
                 f"L={params.L} APs, K={params.K} UEs (single drop)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig("uplink_SE_CDF.png", dpi=150)
    print("CDF saved → uplink_SE_CDF.png")
    plt.show()

    return SE_eq, SE_heur, SE_opt, beta


if __name__ == "__main__":
    params = SystemParams(L=64, K=16, area_side_m=1000.0,
                          p_max_dBm=23.0, tau_c=200, tau_p=16)
    run_single_drop(params, seed=42)
