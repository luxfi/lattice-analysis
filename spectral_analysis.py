#!/usr/bin/env python3
"""
Reciprocal-Space Spectral Analysis for LUX Lattice Crypto
Dragonfire Research Rig v0.1

Takes CSV output from lattice_analysis.go and performs:
1. Distribution comparison (direct vs reciprocal space)
2. Power spectral density analysis
3. Autocorrelation detection (structure in noise = vulnerability)
4. Timing variance analysis for side-channel detection
5. Generates HTML report with inline charts

USAGE:
    pip install numpy scipy matplotlib
    python3 spectral_analysis.py

INPUT:  results/ directory (from Go harness)
OUTPUT: results/report.html — full analysis report
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

# Optional imports — degrade gracefully if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Charts will be skipped.")
    print("Install with: pip install matplotlib")

try:
    from scipy import stats, signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Advanced analysis will be limited.")
    print("Install with: pip install scipy")


RESULTS_DIR = Path("results")
REPORT_FILE = RESULTS_DIR / "report.html"
CHART_DIR = RESULTS_DIR / "charts"


def load_csv(filename):
    """Load a two-column CSV (index, value) into numpy array."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping")
        return None
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    return data[:, 1] if data.ndim > 1 else data


def load_summary():
    """Load the JSON summary from the Go harness."""
    filepath = RESULTS_DIR / "summary.json"
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def analyse_distribution(samples, name):
    """Statistical analysis of a sample distribution."""
    if samples is None or len(samples) == 0:
        return {"name": name, "status": "NO DATA"}

    result = {
        "name": name,
        "n": len(samples),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "variance": float(np.var(samples)),
        "skewness": float(stats.skew(samples)) if HAS_SCIPY else "N/A",
        "kurtosis": float(stats.kurtosis(samples)) if HAS_SCIPY else "N/A",
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }

    # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
    if HAS_SCIPY:
        if len(samples) <= 5000:
            stat, p_value = stats.shapiro(samples[:5000])
            result["normality_test"] = "Shapiro-Wilk"
        else:
            stat, p_value = stats.normaltest(samples)
            result["normality_test"] = "D'Agostino-Pearson"
        result["normality_stat"] = float(stat)
        result["normality_p"] = float(p_value)
        result["is_gaussian"] = p_value > 0.05

    return result


def analyse_spectral(samples, name):
    """Power spectral density analysis — looking for structure in noise."""
    if samples is None or len(samples) < 64:
        return {"name": name, "status": "INSUFFICIENT DATA"}

    result = {"name": name}

    # Compute PSD using Welch's method
    if HAS_SCIPY:
        freqs, psd = signal.welch(samples, nperseg=min(256, len(samples)))
        result["psd_mean"] = float(np.mean(psd))
        result["psd_max"] = float(np.max(psd))
        result["psd_peak_freq"] = float(freqs[np.argmax(psd)])

        # Flatness — for true white noise, PSD should be flat
        # Spectral flatness = geometric mean / arithmetic mean
        psd_positive = psd[psd > 0]
        if len(psd_positive) > 0:
            geo_mean = np.exp(np.mean(np.log(psd_positive)))
            arith_mean = np.mean(psd_positive)
            flatness = geo_mean / arith_mean if arith_mean > 0 else 0
            result["spectral_flatness"] = float(flatness)
            # Flatness of 1.0 = perfectly white noise
            # Flatness < 0.5 = significant spectral structure
            result["is_white_noise"] = flatness > 0.7
        else:
            result["spectral_flatness"] = 0
            result["is_white_noise"] = False

    # Autocorrelation — structure in noise shows up as non-zero autocorrelation
    autocorr = np.correlate(samples - np.mean(samples), samples - np.mean(samples), mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr  # Normalise

    # Check if autocorrelation decays to zero (it should for good noise)
    # Look at lag 1-10: should all be near zero
    if len(autocorr) > 10:
        lag_values = autocorr[1:11]
        max_autocorr = float(np.max(np.abs(lag_values)))
        result["max_autocorrelation_lag1_10"] = max_autocorr
        result["has_structure"] = max_autocorr > 0.05  # 5% threshold

    return result


def analyse_timing(timings, name):
    """Timing analysis for side-channel detection."""
    if timings is None or len(timings) == 0:
        return {"name": name, "status": "NO DATA"}

    result = {
        "name": name,
        "n": len(timings),
        "mean_ns": float(np.mean(timings)),
        "std_ns": float(np.std(timings)),
        "cv": float(np.std(timings) / np.mean(timings)) if np.mean(timings) > 0 else 0,
        "min_ns": float(np.min(timings)),
        "max_ns": float(np.max(timings)),
        "range_ns": float(np.max(timings) - np.min(timings)),
    }

    # Bimodality check — timing that clusters into two groups
    # suggests input-dependent branching
    if HAS_SCIPY:
        # Check if distribution is bimodal using dip test approximation
        hist, bin_edges = np.histogram(timings, bins=50)
        peaks = signal.find_peaks(hist, height=len(timings)*0.01)[0]
        result["num_timing_peaks"] = len(peaks)
        result["is_bimodal"] = len(peaks) >= 2

    # Verdict
    cv = result["cv"]
    if cv < 0.05:
        result["verdict"] = "GOOD — timing appears constant"
    elif cv < 0.1:
        result["verdict"] = "ACCEPTABLE — low variance"
    elif cv < 0.3:
        result["verdict"] = "WARNING — timing variance may be exploitable"
    else:
        result["verdict"] = "CRITICAL — non-constant-time, likely exploitable"

    return result


# ================================================================
# CHART GENERATION
# ================================================================

def generate_charts(direct_samples, ntt_samples):
    """Generate visualisation charts."""
    if not HAS_MATPLOTLIB:
        return {}

    os.makedirs(CHART_DIR, exist_ok=True)
    charts = {}

    # 1. Direct vs Reciprocal space histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Noise Distribution: Direct Space vs Reciprocal Space (NTT)', fontsize=14)

    if direct_samples is not None:
        ax1.hist(direct_samples, bins=80, density=True, alpha=0.7, color='#2196F3', label='Samples')
        # Overlay theoretical Gaussian
        x = np.linspace(np.min(direct_samples), np.max(direct_samples), 200)
        ax1.plot(x, stats.norm.pdf(x, np.mean(direct_samples), np.std(direct_samples)),
                 'r-', linewidth=2, label='Expected Gaussian') if HAS_SCIPY else None
        ax1.set_title('Direct Space (Gaussian Sampler Output)')
        ax1.set_xlabel('Sample Value')
        ax1.set_ylabel('Density')
        ax1.legend()

    if ntt_samples is not None:
        ax2.hist(ntt_samples, bins=80, density=True, alpha=0.7, color='#FF5722', label='NTT Samples')
        if HAS_SCIPY:
            x = np.linspace(np.min(ntt_samples), np.max(ntt_samples), 200)
            ax2.plot(x, stats.norm.pdf(x, np.mean(ntt_samples), np.std(ntt_samples)),
                     'r-', linewidth=2, label='Expected Gaussian')
        ax2.set_title('Reciprocal Space (After NTT Transform)')
        ax2.set_xlabel('Frequency Domain Value')
        ax2.set_ylabel('Density')
        ax2.legend()

    plt.tight_layout()
    chart_path = CHART_DIR / "distribution_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    charts["distribution"] = str(chart_path)

    # 2. Power Spectral Density
    if HAS_SCIPY and direct_samples is not None and len(direct_samples) >= 64:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Power Spectral Density — Detecting Structure in Noise', fontsize=14)

        freqs_d, psd_d = signal.welch(direct_samples, nperseg=min(256, len(direct_samples)))
        ax1.semilogy(freqs_d, psd_d, color='#2196F3')
        ax1.set_title('Direct Space PSD')
        ax1.set_xlabel('Normalised Frequency')
        ax1.set_ylabel('Power/Frequency (dB)')
        ax1.axhline(y=np.mean(psd_d), color='r', linestyle='--', alpha=0.5, label='Mean')
        ax1.legend()

        if ntt_samples is not None and len(ntt_samples) >= 64:
            freqs_r, psd_r = signal.welch(ntt_samples, nperseg=min(256, len(ntt_samples)))
            ax2.semilogy(freqs_r, psd_r, color='#FF5722')
            ax2.set_title('Reciprocal Space PSD')
            ax2.set_xlabel('Normalised Frequency')
            ax2.set_ylabel('Power/Frequency (dB)')
            ax2.axhline(y=np.mean(psd_r), color='r', linestyle='--', alpha=0.5, label='Mean')
            ax2.legend()

        plt.tight_layout()
        chart_path = CHART_DIR / "psd_analysis.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        charts["psd"] = str(chart_path)

    # 3. Autocorrelation comparison
    if direct_samples is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Autocorrelation — Structure Detection', fontsize=14)

        max_lag = min(100, len(direct_samples) // 2)
        acf_d = np.correlate(direct_samples - np.mean(direct_samples),
                             direct_samples - np.mean(direct_samples), mode='full')
        acf_d = acf_d[len(acf_d)//2:len(acf_d)//2 + max_lag]
        acf_d = acf_d / acf_d[0] if acf_d[0] != 0 else acf_d

        ax1.bar(range(max_lag), acf_d, color='#2196F3', alpha=0.7)
        ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
        ax1.axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Direct Space Autocorrelation')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Correlation')
        ax1.legend()

        if ntt_samples is not None:
            acf_r = np.correlate(ntt_samples - np.mean(ntt_samples),
                                 ntt_samples - np.mean(ntt_samples), mode='full')
            acf_r = acf_r[len(acf_r)//2:len(acf_r)//2 + min(max_lag, len(ntt_samples)//2)]
            acf_r = acf_r / acf_r[0] if acf_r[0] != 0 else acf_r

            ax2.bar(range(len(acf_r)), acf_r, color='#FF5722', alpha=0.7)
            ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
            ax2.axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
            ax2.set_title('Reciprocal Space Autocorrelation')
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Correlation')
            ax2.legend()

        plt.tight_layout()
        chart_path = CHART_DIR / "autocorrelation.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        charts["autocorrelation"] = str(chart_path)

    return charts


# ================================================================
# REPORT GENERATION
# ================================================================

def generate_report(summary, analyses, charts):
    """Generate HTML report with findings."""

    findings_html = ""
    for a in analyses:
        severity = "info"
        if isinstance(a.get("verdict"), str):
            if "CRITICAL" in a["verdict"]:
                severity = "critical"
            elif "WARNING" in a["verdict"]:
                severity = "warning"
            elif "GOOD" in a["verdict"]:
                severity = "good"

        findings_html += f"""
        <div class="finding {severity}">
            <h3>{a.get('name', 'Unknown')}</h3>
            <pre>{json.dumps(a, indent=2, default=str)}</pre>
        </div>
        """

    charts_html = ""
    for name, path in charts.items():
        rel_path = os.path.relpath(path, RESULTS_DIR)
        charts_html += f'<div class="chart"><h3>{name.replace("_", " ").title()}</h3><img src="{rel_path}" /></div>\n'

    go_findings = ""
    if summary and "findings" in summary:
        for f in summary["findings"]:
            go_findings += f"<li>{f}</li>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LUX Lattice — Reciprocal-Space Security Analysis</title>
<style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #0a0a0a; color: #e0e0e0; }}
    h1 {{ color: #ff5722; border-bottom: 2px solid #ff5722; padding-bottom: 10px; }}
    h2 {{ color: #2196f3; margin-top: 40px; }}
    h3 {{ color: #ccc; }}
    .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 30px; border-radius: 8px; margin-bottom: 30px; }}
    .header h1 {{ margin: 0; border: none; }}
    .header p {{ color: #888; margin: 5px 0; }}
    .finding {{ padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #666; background: #1a1a1a; }}
    .finding.critical {{ border-left-color: #f44336; background: #1a0a0a; }}
    .finding.warning {{ border-left-color: #ff9800; background: #1a1500; }}
    .finding.good {{ border-left-color: #4caf50; background: #0a1a0a; }}
    .finding.info {{ border-left-color: #2196f3; }}
    .finding pre {{ background: #111; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px; color: #aaa; }}
    .chart {{ margin: 20px 0; text-align: center; }}
    .chart img {{ max-width: 100%; border-radius: 6px; border: 1px solid #333; }}
    .go-findings {{ background: #111; padding: 15px; border-radius: 6px; }}
    .go-findings li {{ margin: 8px 0; }}
    .methodology {{ background: #111; padding: 20px; border-radius: 6px; margin: 20px 0; }}
    .methodology p {{ color: #999; line-height: 1.6; }}
    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #333; color: #666; font-size: 12px; }}
</style>
</head>
<body>

<div class="header">
    <h1>LUX Lattice — Reciprocal-Space Security Analysis</h1>
    <p>Dragonfire Research | {summary.get('timestamp', 'N/A') if summary else 'N/A'}</p>
    <p>Ring Degree: {summary.get('ring_degree', 'N/A') if summary else 'N/A'} | Samples: {summary.get('num_samples', 'N/A') if summary else 'N/A'}</p>
</div>

<div class="methodology">
    <h2>Methodology: Reciprocal-Space Analysis</h2>
    <p>Lattice-based cryptographic security depends on the hardness of finding short vectors in
    high-dimensional lattices. The noise added during encryption (from a Gaussian distribution)
    is what makes the lattice problem hard. However, when this noise is transformed into the
    Number Theoretic Transform (NTT) domain — the discrete analogue of reciprocal space in
    solid-state physics — structural anomalies may become visible that are hidden in direct space.</p>

    <p>This analysis examines the LUX lattice implementation by:</p>
    <p>1. Capturing raw Gaussian noise samples from the sampler (direct space)<br>
    2. Applying NTT transformation to move into reciprocal space<br>
    3. Comparing statistical properties across both domains<br>
    4. Computing power spectral density to detect hidden periodicity<br>
    5. Measuring autocorrelation to identify exploitable structure<br>
    6. Timing crypto operations to detect side-channel leakage</p>

    <p>The physical analogy: just as phonon dispersion reveals defects in a crystal lattice that
    are invisible to direct inspection, spectral analysis of cryptographic noise can reveal
    implementation weaknesses that are invisible to standard statistical tests.</p>
</div>

<h2>Go Harness Findings</h2>
<div class="go-findings">
    <ul>{go_findings}</ul>
</div>

<h2>Spectral Analysis Results</h2>
{charts_html}

<h2>Detailed Findings</h2>
{findings_html}

<div class="footer">
    <p>Generated by Dragonfire Research Rig v0.1 | LUX Lattice Reciprocal-Space Analysis</p>
    <p>This report analyses placeholder data unless the Go harness has been integrated with the actual LUX lattice library.</p>
</div>

</body>
</html>"""

    with open(REPORT_FILE, 'w') as f:
        f.write(html)

    print(f"\n  Report written to {REPORT_FILE}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  LUX Lattice — Reciprocal-Space Spectral Analysis   ║")
    print("║  Dragonfire Research Rig v0.1 (Python)              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR}/ not found. Run lattice_analysis.go first.")
        sys.exit(1)

    # Load data
    print("▸ Loading data...")
    direct_samples = load_csv("noise_samples_direct.csv")
    ntt_samples = load_csv("noise_samples_ntt.csv")
    summary = load_summary()

    # Run analyses
    print("▸ Running statistical analysis...")
    analyses = []

    if direct_samples is not None:
        analyses.append(analyse_distribution(direct_samples, "Direct Space Distribution"))
        analyses.append(analyse_spectral(direct_samples, "Direct Space Spectral"))

    if ntt_samples is not None:
        analyses.append(analyse_distribution(ntt_samples, "Reciprocal Space Distribution"))
        analyses.append(analyse_spectral(ntt_samples, "Reciprocal Space Spectral"))

    # Cross-domain comparison
    if direct_samples is not None and ntt_samples is not None:
        cross = {
            "name": "Cross-Domain Comparison",
            "direct_variance": float(np.var(direct_samples)),
            "reciprocal_variance": float(np.var(ntt_samples)),
            "variance_ratio": float(np.var(direct_samples) / np.var(ntt_samples)) if np.var(ntt_samples) > 0 else "INF",
        }
        if isinstance(cross["variance_ratio"], float) and (cross["variance_ratio"] > 10 or cross["variance_ratio"] < 0.1):
            cross["verdict"] = "WARNING: Significant energy redistribution between domains"
        else:
            cross["verdict"] = "OK: Variance ratio within expected bounds"
        analyses.append(cross)

    # Generate charts
    print("▸ Generating charts...")
    charts = generate_charts(direct_samples, ntt_samples)

    # Generate report
    print("▸ Generating report...")
    generate_report(summary, analyses, charts)

    print()
    print("═══════════════════════════════════════════════════════")
    print("Analysis complete. Open results/report.html in a browser.")
    print("═══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
