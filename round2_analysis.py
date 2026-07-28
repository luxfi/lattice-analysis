#!/usr/bin/env python3
"""
Round 2 Analysis — KS tests, chi-square, and constant discrimination.
Determines whether NTT-domain noise is uniform, and which constant fits.

Run after: go run round2_tests.go

Constants competing:
  -6/5     = -1.200000  (uniform distribution excess kurtosis)
  -√6/2    = -1.224745  (rhombic dodecahedron / BCC Brillouin zone)
  -√1.25   = -1.118034  (Sandreckoner vector)
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("ERROR: scipy required. pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

RESULTS_DIR = Path("results")
CHART_DIR = RESULTS_DIR / "charts"

# Constants
NEG_6_5 = -6/5           # -1.200000
NEG_SQRT6_2 = -np.sqrt(6)/2  # -1.224745
NEG_SQRT125 = -np.sqrt(1.25)  # -1.118034


def test_a_ks():
    """Test A: KS test — is NTT output uniform or Gaussian?"""
    print("═══════════════════════════════════════════════════════════")
    print("TEST A: Kolmogorov-Smirnov Distribution Tests")
    print("═══════════════════════════════════════════════════════════")
    print()

    results = []
    for N in [256, 512, 1024, 2048, 4096]:
        filepath = RESULTS_DIR / f"ks_ntt_N{N}.csv"
        if not filepath.exists():
            print(f"  SKIP N={N}: {filepath} not found")
            continue

        data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str)
        normalised = data[:, 1].astype(float)  # Already in [0,1]
        signed = data[:, 2].astype(float)

        # KS test against uniform [0,1]
        ks_u_stat, ks_u_p = stats.kstest(normalised, 'uniform')

        # KS test against Gaussian (fitted)
        ks_n_stat, ks_n_p = stats.kstest(
            normalised, 'norm',
            args=(np.mean(normalised), np.std(normalised))
        )

        # Chi-square uniformity test (Test F)
        n_bins = 100
        observed, _ = np.histogram(normalised, bins=n_bins, range=(0, 1))
        expected = np.full(n_bins, len(normalised) / n_bins)
        chi2_stat, chi2_p = stats.chisquare(observed, expected)

        # Actual kurtosis
        kurt = float(stats.kurtosis(signed))

        result = {
            "N": N,
            "n_samples": len(normalised),
            "kurtosis": kurt,
            "ks_uniform_stat": float(ks_u_stat),
            "ks_uniform_p": float(ks_u_p),
            "ks_normal_stat": float(ks_n_stat),
            "ks_normal_p": float(ks_n_p),
            "chi2_stat": float(chi2_stat),
            "chi2_p": float(chi2_p),
        }
        results.append(result)

        uniform_verdict = "UNIFORM" if ks_u_p > 0.05 else "NOT uniform"
        normal_verdict = "GAUSSIAN" if ks_n_p > 0.05 else "NOT Gaussian"

        print(f"  N={N:5d} ({len(normalised):6d} samples):")
        print(f"    KS uniform: stat={ks_u_stat:.4f}  p={ks_u_p:.4e}  → {uniform_verdict}")
        print(f"    KS normal:  stat={ks_n_stat:.4f}  p={ks_n_p:.4e}  → {normal_verdict}")
        print(f"    Chi²:       stat={chi2_stat:.1f}  p={chi2_p:.4e}")
        print(f"    Kurtosis:   {kurt:.4f}")
        print()

    return results


def analyse_round2():
    """Analyse the Go round2 results."""
    filepath = RESULTS_DIR / "round2_results.json"
    if not filepath.exists():
        print("ERROR: round2_results.json not found. Run go harness first.")
        return None

    with open(filepath) as f:
        data = json.load(f)

    aggs = data.get("aggregates", [])
    if not aggs:
        print("No aggregate results found.")
        return data

    # ── Test B: Sigma variation ──
    print("═══════════════════════════════════════════════════════════")
    print("TEST B: Kurtosis vs Gaussian Sigma")
    print("═══════════════════════════════════════════════════════════")
    print()
    print(f"  {'Sigma':>8s}  {'Mean κ':>10s}  {'Std':>8s}  {'Δ(-6/5)':>10s}  {'Δ(-√6/2)':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}")
    b_results = [a for a in aggs if a['test'] == 'B']
    for a in b_results:
        d65 = a['mean_ntt_kurtosis'] - NEG_6_5
        dsqrt6 = a['mean_ntt_kurtosis'] - NEG_SQRT6_2
        print(f"  {a['parameter']:>8s}  {a['mean_ntt_kurtosis']:>10.4f}  {a['std_ntt_kurtosis']:>8.4f}  {d65:>+10.4f}  {dsqrt6:>+10.4f}")
    print()

    # ── Test C: Bound variation ──
    print("═══════════════════════════════════════════════════════════")
    print("TEST C: Bounded vs Unbounded")
    print("═══════════════════════════════════════════════════════════")
    print()
    print(f"  {'Bound':>10s}  {'Mean κ':>10s}  {'Std':>8s}  {'Δ(-6/5)':>10s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}")
    c_results = [a for a in aggs if a['test'] == 'C']
    for a in c_results:
        d65 = a['mean_ntt_kurtosis'] - NEG_6_5
        print(f"  {a['parameter']:>10s}  {a['mean_ntt_kurtosis']:>10.4f}  {a['std_ntt_kurtosis']:>8.4f}  {d65:>+10.4f}")
    print()

    # ── Test D: Modulus variation ──
    print("═══════════════════════════════════════════════════════════")
    print("TEST D: Kurtosis vs Modulus Size")
    print("═══════════════════════════════════════════════════════════")
    print()
    print(f"  {'log₂(q)':>10s}  {'Mean κ':>10s}  {'Std':>8s}  {'Δ(-6/5)':>10s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}")
    d_results = [a for a in aggs if a['test'] == 'D']
    for a in d_results:
        d65 = a['mean_ntt_kurtosis'] - NEG_6_5
        print(f"  {a['parameter']:>10s}  {a['mean_ntt_kurtosis']:>10.4f}  {a['std_ntt_kurtosis']:>8.4f}  {d65:>+10.4f}")
    print()

    # ── Test E: DFT vs NTT ──
    print("═══════════════════════════════════════════════════════════")
    print("TEST E: Plain DFT vs Cyclotomic NTT")
    print("═══════════════════════════════════════════════════════════")
    print()
    e_results = [a for a in aggs if a['test'] == 'E']
    for a in e_results:
        d65 = a['mean_ntt_kurtosis'] - NEG_6_5
        print(f"  {a['label']:>25s}: κ={a['mean_ntt_kurtosis']:.4f}±{a['std_ntt_kurtosis']:.4f}  Δ(-6/5)={d65:+.4f}")
    print()

    return data


def constant_discrimination(data):
    """Statistical test: which constant best fits the data?"""
    print("═══════════════════════════════════════════════════════════")
    print("CONSTANT DISCRIMINATION")
    print("═══════════════════════════════════════════════════════════")
    print()

    all_results = data.get("all_results", [])
    # Get all NTT kurtosis values from tests that use the real NTT
    ntt_kvals = [r['ntt_kurtosis'] for r in all_results
                 if r['test'] in ('A', 'B', 'C', 'D', 'E_NTT')]

    if not ntt_kvals:
        print("  No NTT kurtosis data available.")
        return

    kvals = np.array(ntt_kvals)
    n = len(kvals)
    m = np.mean(kvals)
    se = np.std(kvals) / np.sqrt(n)  # Standard error of mean

    print(f"  Sample size: {n}")
    print(f"  Mean κ: {m:.6f}")
    print(f"  Std error: {se:.6f}")
    print(f"  95% CI: [{m - 1.96*se:.6f}, {m + 1.96*se:.6f}]")
    print()

    constants = {
        "-6/5 (uniform)": NEG_6_5,
        "-√6/2 (rhombic dodec)": NEG_SQRT6_2,
        "-√1.25 (Sandreckoner)": NEG_SQRT125,
    }

    print(f"  {'Constant':>25s}  {'Value':>10s}  {'Δ from mean':>12s}  {'t-stat':>8s}  {'p-value':>10s}  {'Result':>15s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*15}")

    for name, val in constants.items():
        diff = m - val
        t_stat = diff / se if se > 0 else float('inf')
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))  # Two-tailed

        if p_val > 0.05:
            result = "CONSISTENT"
        else:
            result = "REJECTED"

        print(f"  {name:>25s}  {val:>10.6f}  {diff:>+12.6f}  {t_stat:>8.2f}  {p_val:>10.4e}  {result:>15s}")

    print()


def generate_charts(data):
    """Generate summary charts."""
    if not HAS_MPL or data is None:
        return

    os.makedirs(CHART_DIR, exist_ok=True)
    aggs = data.get("aggregates", [])

    # Chart 1: Sigma sweep
    b_results = [a for a in aggs if a['test'] == 'B']
    if b_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        sigmas = [float(a['parameter'].split('=')[1]) for a in b_results]
        kvals = [a['mean_ntt_kurtosis'] for a in b_results]
        stds = [a['std_ntt_kurtosis'] for a in b_results]

        ax.errorbar(sigmas, kvals, yerr=stds, fmt='o-', color='#2196F3',
                    capsize=5, markersize=8, label='Measured κ')
        ax.axhline(y=NEG_6_5, color='#F44336', linestyle='--', linewidth=2, label=f'-6/5 = {NEG_6_5:.4f}')
        ax.axhline(y=NEG_SQRT6_2, color='#4CAF50', linestyle=':', linewidth=2, label=f'-√6/2 = {NEG_SQRT6_2:.4f}')
        ax.axhline(y=NEG_SQRT125, color='#FF9800', linestyle='-.', linewidth=2, label=f'-√1.25 = {NEG_SQRT125:.4f}')
        ax.set_xlabel('Gaussian σ', fontsize=12)
        ax.set_ylabel('NTT-domain Excess Kurtosis', fontsize=12)
        ax.set_title('Test B: NTT Kurtosis vs Gaussian Sigma\n(Does sigma affect the distribution?)', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xscale('log')
        plt.savefig(CHART_DIR / "round2_sigma_sweep.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Chart 2: Bound sweep
    c_results = [a for a in aggs if a['test'] == 'C']
    if c_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        bounds = [float(a['parameter'].split('=')[1].rstrip('σ')) for a in c_results]
        kvals = [a['mean_ntt_kurtosis'] for a in c_results]
        stds = [a['std_ntt_kurtosis'] for a in c_results]

        ax.errorbar(bounds, kvals, yerr=stds, fmt='s-', color='#9C27B0',
                    capsize=5, markersize=8, label='Measured κ')
        ax.axhline(y=NEG_6_5, color='#F44336', linestyle='--', linewidth=2, label=f'-6/5 = {NEG_6_5:.4f}')
        ax.axhline(y=NEG_SQRT6_2, color='#4CAF50', linestyle=':', linewidth=2, label=f'-√6/2 = {NEG_SQRT6_2:.4f}')
        ax.set_xlabel('Rejection Bound (multiples of σ)', fontsize=12)
        ax.set_ylabel('NTT-domain Excess Kurtosis', fontsize=12)
        ax.set_title('Test C: Bounded vs Unbounded Sampling\n(Does truncation cause uniformity?)', fontsize=13)
        ax.legend(fontsize=10)
        plt.savefig(CHART_DIR / "round2_bound_sweep.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Chart 3: Modulus sweep
    d_results = [a for a in aggs if a['test'] == 'D']
    if d_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        logqs = [int(a['parameter'].split('=')[1]) for a in d_results]
        kvals = [a['mean_ntt_kurtosis'] for a in d_results]
        stds = [a['std_ntt_kurtosis'] for a in d_results]

        ax.errorbar(logqs, kvals, yerr=stds, fmt='D-', color='#FF5722',
                    capsize=5, markersize=8, label='Measured κ')
        ax.axhline(y=NEG_6_5, color='#F44336', linestyle='--', linewidth=2, label=f'-6/5 = {NEG_6_5:.4f}')
        ax.axhline(y=NEG_SQRT6_2, color='#4CAF50', linestyle=':', linewidth=2, label=f'-√6/2 = {NEG_SQRT6_2:.4f}')
        ax.set_xlabel('log₂(q) — Modulus Bit Size', fontsize=12)
        ax.set_ylabel('NTT-domain Excess Kurtosis', fontsize=12)
        ax.set_title('Test D: NTT Kurtosis vs Modulus Size\n(Does mod reduction cause uniformity?)', fontsize=13)
        ax.legend(fontsize=10)
        plt.savefig(CHART_DIR / "round2_modulus_sweep.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Chart 4: All constants comparison
    all_results = data.get("all_results", [])
    ntt_kvals = [r['ntt_kurtosis'] for r in all_results
                 if r['test'] in ('A', 'B', 'C', 'D', 'E_NTT')]
    if ntt_kvals:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ntt_kvals, bins=40, density=True, alpha=0.7, color='#2196F3', label='Measured κ')
        ax.axvline(x=NEG_6_5, color='#F44336', linestyle='--', linewidth=2.5, label=f'-6/5 = {NEG_6_5:.4f}')
        ax.axvline(x=NEG_SQRT6_2, color='#4CAF50', linestyle=':', linewidth=2.5, label=f'-√6/2 = {NEG_SQRT6_2:.4f}')
        ax.axvline(x=NEG_SQRT125, color='#FF9800', linestyle='-.', linewidth=2.5, label=f'-√1.25 = {NEG_SQRT125:.4f}')
        ax.axvline(x=np.mean(ntt_kvals), color='white', linestyle='-', linewidth=1.5, label=f'Mean = {np.mean(ntt_kvals):.4f}')
        ax.set_xlabel('Excess Kurtosis', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of Measured NTT Kurtosis (n={len(ntt_kvals)} trials)\nvs Candidate Constants', fontsize=13)
        ax.legend(fontsize=10)
        plt.savefig(CHART_DIR / "round2_constant_discrimination.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Charts saved to {CHART_DIR}/round2_*.png")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Round 2 Analysis — Distribution Tests & Discrimination  ║")
    print("║  Dragonfire / In2Infinity Research                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Test A: KS
    ks_results = test_a_ks()
    print()

    # Analyse Go results
    data = analyse_round2()
    print()

    # Constant discrimination
    if data:
        constant_discrimination(data)

    # Charts
    if data:
        generate_charts(data)

    print()
    print("═══════════════════════════════════════════════════════════")
    print("ROUND 2 ANALYSIS COMPLETE")
    print("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
