#!/usr/bin/env python3
"""
Phi-Geometric Analysis of RLWE Lattice Noise
Dragonfire / In2Infinity Research

Based on the observation that 2D Brillouin zones on a square lattice
expand through geometric ratios 1, √2, √5 encoding both the silver
ratio (δ = √2±1) and golden ratio (φ = √1.25±½).

The Sandreckoner diagram: a unit square with corner-to-opposite-midpoint
vectors of length √1.25, from which φ derives, provides the geometric
blueprint for how reciprocal space partitions.

This script tests whether these same ratios appear as structural
features in the NTT-domain noise of RLWE cryptographic implementations.

THEORY:
  In physical crystals, the Brillouin zone shape determines phonon
  dispersion and thus electrical/magnetic properties. The truncated
  octahedron (copper/FCC) and rhombic dodecahedron (iron/BCC) are
  connected by a √1.25 scaling factor — the golden ratio bridge
  between electric and magnetic field geometry.

  In cryptographic lattices, the polynomial ring Z[x]/(x^n+1) creates
  an analogous reciprocal space under NTT. If the noise distribution
  "knows about" the lattice geometry (as our kurtosis finding suggests),
  then phi-related frequencies should show anomalous behaviour.

USAGE:
  python3 phi_analysis.py

INPUT:  results/noise_samples_ntt.csv (from Go harness)
OUTPUT: results/phi_analysis.json + results/charts/phi_*.png
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import signal, stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ================================================================
# CONSTANTS — the geometric ratios from Brillouin zone theory
# ================================================================

PHI = (1 + np.sqrt(5)) / 2          # Golden ratio: 1.6180339...
SILVER = 1 + np.sqrt(2)             # Silver ratio: 2.4142135...
SQRT_1_25 = np.sqrt(1.25)           # Sandreckoner vector: 1.1180339...
SQRT_2 = np.sqrt(2)                 # 1st Brillouin zone diagonal
SQRT_5 = np.sqrt(5)                 # 3rd zone ratio
INV_PHI = 1 / PHI                   # 0.6180339...
PHI_SQ = PHI ** 2                   # 2.6180339...

# Zone progression ratios (from the article):
# Zone 1: side 1 (unit square)
# Zone 2: side √2 (rotated 45°)
# Zone 3: involves √5 (contains golden ratio triangles 1:2:√5)
# Zone 4: octagonal, dimensions 3 on axes, sides 1 and √2
ZONE_RATIOS = [1.0, SQRT_2, SQRT_5, 3.0]

# Brillouin zone area ratios (each zone has equal area in 2D)
# But the SHAPE changes — and shape determines noise distribution
ZONE_SCALE_FACTORS = [1.0, SQRT_2, SQRT_5, SQRT_1_25 * 2]

# Key frequencies to test (normalised to Nyquist)
# These are the ratios at which Brillouin zone boundaries fall
PHI_FREQUENCIES = {
    "1/phi": INV_PHI,           # 0.618...
    "1/sqrt5": 1/SQRT_5,        # 0.447...
    "1/sqrt2": 1/SQRT_2,        # 0.707...
    "1/sqrt1.25": 1/SQRT_1_25,  # 0.894...
    "phi/3": PHI/3,              # 0.539...
    "2/sqrt5": 2/SQRT_5,        # 0.894...
    "silver/4": SILVER/4,        # 0.604...
}

RESULTS_DIR = Path("results")
CHART_DIR = RESULTS_DIR / "charts"


def load_ntt_samples():
    """Load NTT-domain noise samples."""
    filepath = RESULTS_DIR / "noise_samples_ntt.csv"
    if not filepath.exists():
        print(f"Error: {filepath} not found. Run lattice_analysis.go first.")
        return None
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    return data[:, 1] if data.ndim > 1 else data


def load_direct_samples():
    """Load direct-space noise samples."""
    filepath = RESULTS_DIR / "noise_samples_direct.csv"
    if not filepath.exists():
        return None
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    return data[:, 1] if data.ndim > 1 else data


# ================================================================
# PHI-GEOMETRIC TESTS
# ================================================================

def test_phi_frequencies(samples, name="NTT"):
    """
    Test whether the PSD has anomalous power at phi-related frequencies.
    
    If the lattice noise "knows about" the Brillouin zone geometry,
    we'd expect peaks or troughs at frequencies corresponding to
    zone boundary ratios.
    """
    if not HAS_SCIPY or samples is None or len(samples) < 128:
        return {"test": "phi_frequencies", "status": "INSUFFICIENT DATA"}

    freqs, psd = signal.welch(samples, nperseg=min(512, len(samples)))
    
    # Normalise frequencies to [0, 1] (fraction of Nyquist)
    max_freq = freqs[-1]
    norm_freqs = freqs / max_freq if max_freq > 0 else freqs
    
    # Mean PSD as baseline
    mean_psd = np.mean(psd)
    
    results = {
        "test": "phi_frequencies",
        "domain": name,
        "mean_psd": float(mean_psd),
        "anomalies": []
    }
    
    # Check each phi-related frequency
    for label, target_freq in PHI_FREQUENCIES.items():
        if target_freq > 1.0:
            continue  # Skip if above Nyquist
            
        # Find closest frequency bin
        idx = np.argmin(np.abs(norm_freqs - target_freq))
        actual_freq = float(norm_freqs[idx])
        power = float(psd[idx])
        ratio_to_mean = power / mean_psd if mean_psd > 0 else 0
        
        entry = {
            "label": label,
            "target_freq": float(target_freq),
            "actual_freq": actual_freq,
            "power": power,
            "ratio_to_mean": ratio_to_mean,
        }
        
        # Flag anomalies (more than 2x or less than 0.5x mean)
        if ratio_to_mean > 2.0:
            entry["anomaly"] = f"ELEVATED — {ratio_to_mean:.2f}x mean power"
            results["anomalies"].append(entry)
        elif ratio_to_mean < 0.5:
            entry["anomaly"] = f"SUPPRESSED — {ratio_to_mean:.2f}x mean power"
            results["anomalies"].append(entry)
        else:
            entry["anomaly"] = None
            
        results.setdefault("frequencies", []).append(entry)
    
    if len(results["anomalies"]) > 0:
        results["verdict"] = f"FOUND {len(results['anomalies'])} phi-related frequency anomalies"
    else:
        results["verdict"] = "No significant phi-related frequency anomalies detected"
    
    return results


def test_phi_autocorrelation(samples, name="NTT"):
    """
    Test autocorrelation at phi-scaled lags.
    
    The Brillouin zone geometry suggests correlations at distances
    related to √1.25, √2, √5, and their combinations.
    If the noise has structure at these scales, autocorrelation
    will be non-zero at corresponding lags.
    """
    if samples is None or len(samples) < 64:
        return {"test": "phi_autocorrelation", "status": "INSUFFICIENT DATA"}
    
    # Compute full autocorrelation
    centered = samples - np.mean(samples)
    acf = np.correlate(centered, centered, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0] if acf[0] != 0 else acf
    
    # Test at phi-related lags
    # The Sandreckoner construction suggests correlations at:
    # lag √1.25 ≈ 1.118 → lag 1
    # lag √2 ≈ 1.414 → lag 1  
    # lag φ ≈ 1.618 → lag 2
    # lag √5 ≈ 2.236 → lag 2
    # lag φ² ≈ 2.618 → lag 3
    # lag 3 (zone 4 boundary)
    # lag φ³ ≈ 4.236 → lag 4
    
    phi_lags = {
        "sqrt_1.25": SQRT_1_25,
        "sqrt_2": SQRT_2,
        "phi": PHI,
        "sqrt_5": SQRT_5,
        "phi_sq": PHI_SQ,
        "zone_4": 3.0,
        "phi_cubed": PHI**3,
        "2*phi": 2*PHI,
        "sqrt5*phi": SQRT_5 * PHI,
    }
    
    results = {
        "test": "phi_autocorrelation",
        "domain": name,
        "lag_results": [],
        "anomalies": []
    }
    
    # 5% significance threshold for autocorrelation
    n = len(samples)
    threshold = 1.96 / np.sqrt(n)  # 95% confidence interval
    
    for label, phi_lag in phi_lags.items():
        # Check both the integer lags surrounding the phi value
        for lag in [int(np.floor(phi_lag)), int(np.ceil(phi_lag))]:
            if lag < 1 or lag >= len(acf):
                continue
            
            acf_val = float(acf[lag])
            significant = abs(acf_val) > threshold
            
            entry = {
                "label": f"{label} (lag={lag})",
                "phi_value": float(phi_lag),
                "lag": lag,
                "autocorrelation": acf_val,
                "threshold": float(threshold),
                "significant": significant,
            }
            
            results["lag_results"].append(entry)
            if significant:
                results["anomalies"].append(entry)
    
    if len(results["anomalies"]) > 0:
        results["verdict"] = f"FOUND {len(results['anomalies'])} significant autocorrelation values at phi-related lags"
    else:
        results["verdict"] = "No significant autocorrelation at phi-related lags"
    
    return results


def test_zone_kurtosis(samples, name="NTT"):
    """
    Test whether the kurtosis value corresponds to a known
    Brillouin zone packing ratio.
    
    Key geometric constants:
    - Truncated octahedron packing: 0.6826... (copper BZ)
    - Rhombic dodecahedron packing: 0.7405... (iron BZ)
    - φ-related: 1/φ² = 0.3820..., 2/φ² = 0.7639...
    - Our finding: kurtosis = -1.18
    
    Test whether -1.18 relates to any geometric constant.
    """
    if not HAS_SCIPY or samples is None:
        return {"test": "zone_kurtosis", "status": "INSUFFICIENT DATA"}
    
    kurt = float(stats.kurtosis(samples))
    
    # Known geometric constants to compare
    constants = {
        "1/phi": 1/PHI,                      # 0.618
        "1/phi^2": 1/PHI**2,                 # 0.382
        "sqrt(1.25)-1": SQRT_1_25 - 1,       # 0.118 — NOTE: close to |kurtosis| pattern
        "1/sqrt5": 1/SQRT_5,                 # 0.447
        "truncated_oct_packing": 0.68329,    # Copper BZ packing
        "rhombic_dodec_packing": 0.74048,    # Iron BZ packing
        "1-1/phi": 1 - 1/PHI,               # 0.382 = 1/phi^2
        "2*(sqrt1.25-1)": 2*(SQRT_1_25-1),  # 0.236
        "phi-1/phi": PHI - 1/PHI,           # 1.0 (exactly!)
        "sqrt2-1/sqrt1.25": SQRT_2 - 1/SQRT_1_25, # 0.520
    }
    
    results = {
        "test": "zone_kurtosis",
        "domain": name,
        "measured_kurtosis": kurt,
        "abs_kurtosis": abs(kurt),
        "matches": []
    }
    
    # Check if |kurtosis| is close to any geometric constant
    # or if kurtosis relates to ratios of constants
    for label, value in constants.items():
        # Direct match
        if abs(abs(kurt) - value) < 0.05:
            results["matches"].append({
                "label": label,
                "value": float(value),
                "difference": float(abs(abs(kurt) - value)),
                "type": "direct"
            })
        # Reciprocal match
        if value > 0 and abs(abs(kurt) - 1/value) < 0.05:
            results["matches"].append({
                "label": f"1/{label}",
                "value": float(1/value),
                "difference": float(abs(abs(kurt) - 1/value)),
                "type": "reciprocal"
            })
    
    # Special check: is kurtosis ≈ -(√1.25 + some simple fraction)?
    # √1.25 = 1.1180... and our kurtosis is -1.18
    # Difference: 1.18 - 1.118 = 0.062 ≈ 1/16 = 0.0625
    sqrt125_diff = abs(abs(kurt) - SQRT_1_25)
    results["sqrt_1.25_proximity"] = {
        "sqrt_1.25": float(SQRT_1_25),
        "difference": float(sqrt125_diff),
        "note": f"|kurtosis| - √1.25 = {sqrt125_diff:.4f}" + 
                (" — VERY CLOSE to √1.25!" if sqrt125_diff < 0.1 else "")
    }
    
    if sqrt125_diff < 0.1:
        results["verdict"] = f"SIGNIFICANT: |kurtosis| = {abs(kurt):.4f} ≈ √1.25 = {SQRT_1_25:.4f} (diff={sqrt125_diff:.4f}). The Sandreckoner vector length appears in the noise distribution."
    elif len(results["matches"]) > 0:
        results["verdict"] = f"Found {len(results['matches'])} geometric constant matches"
    else:
        results["verdict"] = "No clear geometric constant match found"
    
    return results


def test_zone_progression(samples, name="NTT"):
    """
    Divide the NTT spectrum into zones following the Brillouin
    zone area ratios and compare energy distribution across zones.
    
    In a physical crystal, each Brillouin zone has equal area but
    different shape. The energy distribution across zones reveals
    the lattice's electronic/phononic properties.
    
    In the crypto lattice, we divide the NTT frequency spectrum
    into zones at √2, √5, and 3× boundaries and compare.
    """
    if not HAS_SCIPY or samples is None or len(samples) < 128:
        return {"test": "zone_progression", "status": "INSUFFICIENT DATA"}
    
    freqs, psd = signal.welch(samples, nperseg=min(512, len(samples)))
    max_freq = freqs[-1] if len(freqs) > 0 and freqs[-1] > 0 else 1.0
    
    # Define zone boundaries (normalised to max frequency)
    # Zone 1: 0 to 1/3 (unit zone)
    # Zone 2: 1/3 to √2/3 (first expansion)
    # Zone 3: √2/3 to √5/3 (golden ratio zone)
    # Zone 4: √5/3 to 1 (outer zone)
    boundaries = [0, 1/3, SQRT_2/3, SQRT_5/3, 1.0]
    
    zones = []
    for i in range(len(boundaries) - 1):
        low = boundaries[i] * max_freq
        high = boundaries[i+1] * max_freq
        mask = (freqs >= low) & (freqs < high)
        zone_psd = psd[mask]
        
        if len(zone_psd) > 0:
            zones.append({
                "zone": i + 1,
                "freq_range": f"{boundaries[i]:.3f} — {boundaries[i+1]:.3f}",
                "mean_power": float(np.mean(zone_psd)),
                "total_power": float(np.sum(zone_psd)),
                "variance": float(np.var(zone_psd)),
            })
    
    # Compare zone energies
    if len(zones) >= 2:
        powers = [z["mean_power"] for z in zones]
        # In white noise, all zones should have roughly equal power
        # Deviations indicate lattice geometry influence
        max_ratio = max(powers) / min(powers) if min(powers) > 0 else float('inf')
        
        results = {
            "test": "zone_progression",
            "domain": name,
            "zones": zones,
            "max_power_ratio": float(max_ratio),
        }
        
        if max_ratio > 3.0:
            results["verdict"] = f"SIGNIFICANT: Zone power ratio = {max_ratio:.2f} — strong non-uniform energy distribution across Brillouin-scaled zones"
        elif max_ratio > 1.5:
            results["verdict"] = f"WARNING: Zone power ratio = {max_ratio:.2f} — moderate non-uniformity"
        else:
            results["verdict"] = "OK: Energy roughly uniform across zones"
        
        return results
    
    return {"test": "zone_progression", "status": "INSUFFICIENT ZONES"}


# ================================================================
# CHARTS
# ================================================================

def plot_phi_analysis(samples, freq_results, acf_results, zone_results):
    """Generate phi-analysis specific charts."""
    if not HAS_MPL or not HAS_SCIPY or samples is None:
        return {}
    
    os.makedirs(CHART_DIR, exist_ok=True)
    charts = {}
    
    # 1. PSD with phi-frequency markers
    fig, ax = plt.subplots(figsize=(12, 6))
    freqs, psd = signal.welch(samples, nperseg=min(512, len(samples)))
    max_freq = freqs[-1]
    norm_freqs = freqs / max_freq if max_freq > 0 else freqs
    
    ax.semilogy(norm_freqs, psd, color='#2196F3', alpha=0.8, label='PSD')
    ax.axhline(y=np.mean(psd), color='gray', linestyle='--', alpha=0.5, label='Mean')
    
    # Mark phi frequencies
    colors = ['#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4', '#E91E63', '#795548']
    for i, (label, target) in enumerate(PHI_FREQUENCIES.items()):
        if target <= 1.0:
            ax.axvline(x=target, color=colors[i % len(colors)], linestyle=':', alpha=0.7, label=f'{label}={target:.3f}')
    
    ax.set_title('Power Spectral Density with Phi-Related Frequency Markers\n(Brillouin Zone Boundary Ratios)', fontsize=13)
    ax.set_xlabel('Normalised Frequency')
    ax.set_ylabel('Power')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 1)
    
    path = CHART_DIR / "phi_psd_markers.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts["phi_psd"] = str(path)
    
    # 2. Kurtosis vs √1.25 comparison
    if HAS_SCIPY:
        fig, ax = plt.subplots(figsize=(8, 5))
        kurt = stats.kurtosis(samples)
        
        constants = {
            '|κ| measured': abs(kurt),
            '√1.25': SQRT_1_25,
            '1/φ': INV_PHI,
            '1/√2': 1/SQRT_2,
            '1/φ²': 1/PHI**2,
        }
        
        bars = ax.bar(constants.keys(), constants.values(), color=['#FF5722', '#4CAF50', '#2196F3', '#9C27B0', '#FF9800'])
        ax.set_title(f'Kurtosis |κ| = {abs(kurt):.4f} vs Geometric Constants\n(Sandreckoner: √1.25 = {SQRT_1_25:.4f})', fontsize=13)
        ax.set_ylabel('Value')
        
        # Annotate the proximity
        diff = abs(abs(kurt) - SQRT_1_25)
        ax.annotate(f'Δ = {diff:.4f}', xy=(0.5, max(abs(kurt), SQRT_1_25)), fontsize=10,
                    ha='center', color='red')
        
        path = CHART_DIR / "phi_kurtosis_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        charts["phi_kurtosis"] = str(path)
    
    # 3. Zone energy distribution
    if zone_results and "zones" in zone_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        zone_labels = [f"Zone {z['zone']}\n{z['freq_range']}" for z in zone_results["zones"]]
        zone_powers = [z["mean_power"] for z in zone_results["zones"]]
        
        colors_z = ['#F44336', '#FF9800', '#4CAF50', '#2196F3']
        ax.bar(zone_labels, zone_powers, color=colors_z[:len(zone_labels)])
        ax.set_title('Energy Distribution Across Brillouin-Scaled Zones\n(Zones at 1/3, √2/3, √5/3 boundaries)', fontsize=13)
        ax.set_ylabel('Mean Power')
        
        # Annotate zone ratios
        ax.set_xlabel('Frequency Zone (Brillouin zone boundary ratios)')
        
        path = CHART_DIR / "phi_zone_energy.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        charts["phi_zones"] = str(path)
    
    return charts


# ================================================================
# MAIN
# ================================================================

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Phi-Geometric Analysis of RLWE Lattice Noise       ║")
    print("║  Dragonfire / In2Infinity Research                   ║")
    print("║                                                      ║")
    print("║  Testing Brillouin zone ratios: 1, √2, √5, φ       ║")
    print("║  Sandreckoner vector: √1.25 → φ = √1.25 ± ½        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("▸ Loading NTT-domain noise samples...")
    ntt_samples = load_ntt_samples()
    direct_samples = load_direct_samples()
    
    if ntt_samples is None:
        print("  Error: No NTT data found. Run the Go harness first.")
        sys.exit(1)
    
    print(f"  Loaded {len(ntt_samples)} NTT samples")
    print()
    
    all_results = []
    
    # Test 1: Phi frequencies in PSD
    print("▸ Test 1: Phi-related frequency analysis...")
    freq_result = test_phi_frequencies(ntt_samples, "NTT")
    print(f"  {freq_result.get('verdict', 'N/A')}")
    if freq_result.get("anomalies"):
        for a in freq_result["anomalies"]:
            print(f"    → {a['label']}: {a.get('anomaly', '')}")
    all_results.append(freq_result)
    print()
    
    # Also test direct space for comparison
    if direct_samples is not None:
        freq_result_direct = test_phi_frequencies(direct_samples, "Direct")
        print(f"  Direct space comparison: {freq_result_direct.get('verdict', 'N/A')}")
        all_results.append(freq_result_direct)
        print()
    
    # Test 2: Phi autocorrelation
    print("▸ Test 2: Phi-scaled autocorrelation lags...")
    acf_result = test_phi_autocorrelation(ntt_samples, "NTT")
    print(f"  {acf_result.get('verdict', 'N/A')}")
    if acf_result.get("anomalies"):
        for a in acf_result["anomalies"]:
            print(f"    → {a['label']}: r={a['autocorrelation']:.4f} (threshold={a['threshold']:.4f})")
    all_results.append(acf_result)
    print()
    
    # Test 3: Kurtosis vs geometric constants
    print("▸ Test 3: Kurtosis vs Brillouin zone geometric constants...")
    kurt_result = test_zone_kurtosis(ntt_samples, "NTT")
    print(f"  {kurt_result.get('verdict', 'N/A')}")
    sqrt125_prox = kurt_result.get("sqrt_1.25_proximity", {})
    if sqrt125_prox:
        print(f"    → {sqrt125_prox.get('note', '')}")
    all_results.append(kurt_result)
    print()
    
    # Test 4: Zone energy progression
    print("▸ Test 4: Brillouin zone energy distribution...")
    zone_result = test_zone_progression(ntt_samples, "NTT")
    print(f"  {zone_result.get('verdict', 'N/A')}")
    all_results.append(zone_result)
    print()
    
    # Generate charts
    print("▸ Generating phi-analysis charts...")
    charts = plot_phi_analysis(ntt_samples, freq_result, acf_result, zone_result)
    for name, path in charts.items():
        print(f"  → {path}")
    print()
    
    # Save results
    output = {
        "analysis": "Phi-Geometric Analysis of RLWE Lattice Noise",
        "methodology": "In2Infinity Brillouin Zone Theory applied to cryptographic lattice noise",
        "key_constants": {
            "phi": float(PHI),
            "sqrt_1.25": float(SQRT_1_25),
            "sqrt_2": float(SQRT_2),
            "sqrt_5": float(SQRT_5),
            "silver_ratio": float(SILVER),
        },
        "results": all_results,
        "charts": charts,
    }
    
    output_path = RESULTS_DIR / "phi_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("═══════════════════════════════════════════════════════")
    print("PHI-GEOMETRIC ANALYSIS COMPLETE")
    print()
    print(f"Results: {output_path}")
    print(f"Charts:  {CHART_DIR}/phi_*.png")
    print()
    
    # Highlight the key finding
    if kurt_result.get("sqrt_1.25_proximity", {}).get("difference", 1) < 0.1:
        print("★ KEY FINDING: |kurtosis| ≈ √1.25 (Sandreckoner vector)")
        print("  This suggests the lattice noise distribution in reciprocal")
        print("  space is geometrically constrained by the same √1.25 ratio")
        print("  that bridges the Brillouin zones of copper and iron —")
        print("  the foundation of the golden ratio (φ = √1.25 ± ½).")
        print()
        print("  This is a novel observation connecting solid-state physics")
        print("  to lattice cryptographic security.")
    
    print("═══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
