# LUX Lattice Security Analysis Rig
## Dragonfire / In2Infinity Research

Self-contained analysis toolset for the LUX lattice cryptography stack.
Instruments the **real** `luxfi/lattice` library — no mocks, no placeholders.

### Quick Start

```bash
# Prerequisites: Go 1.25+, Python 3 with numpy/scipy/matplotlib

# 1. Clone the LUX repos (if not already done)
cd ~/projects/lux-review
git clone https://github.com/luxfi/lattice.git
git clone https://github.com/luxfi/consensus.git

# 2. Set up the rig
cd rig/
go mod tidy

# 3. Run the full test battery
./run_all.sh
```

### What It Tests

| Tool | Language | What It Does |
|------|----------|--------------|
| `lattice_analysis.go` | Go | Captures Gaussian noise, applies NTT, measures timing (side-channel detection) |
| `kurtosis_sweep.go` | Go | Parameter sweep: kurtosis across N=256-4096, multiple moduli, 20 trials each |
| `round2_tests.go` | Go | Full test battery: KS tests, sigma/bound/modulus variation, DFT vs NTT |
| `spectral_analysis.py` | Python | PSD, autocorrelation, distribution charts, HTML report |
| `phi_analysis.py` | Python | Brillouin zone geometric ratio analysis |
| `round2_analysis.py` | Python | KS/chi-square tests, constant discrimination, summary charts |

### Key Findings (Confirmed)

**CRITICAL: Gaussian sampler timing side-channel (CV=0.376)**
- `lattice/ring/sampler_gaussian.go:138` — rejection sampling creates data-dependent timing
- Mean: 42,700ns, range: 31,178ns to 808,183ns (26x spread)
- NIST dropped Gaussian sampling from Kyber/Dilithium for exactly this reason

**HIGH: NTT transforms Gaussian to uniform**
- Confirmed across 520+ trials, all parameters
- Excess kurtosis = -1.197 (uniform = -1.200)
- KS test confirms: p > 0.05 for uniformity, p ~ 0 for Gaussianity
- This is specific to cyclotomic NTT (plain DFT does not do this)
- Mathematically expected (smoothing lemma) — not a vulnerability, but validates methodology

**HIGH: `bytes.Equal` in witness verification**
- `consensus/protocol/quasar/witness.go:233` — timing oracle on proof comparison

### Running Individual Tests

```bash
# Timing + noise analysis (produces results/ directory)
go run lattice_analysis.go

# Parameter sweep across ring degrees
go run kurtosis_sweep.go

# Full A-E test battery
go run round2_tests.go

# Python analysis (run after Go harnesses)
python3 spectral_analysis.py      # Charts + HTML report
python3 phi_analysis.py           # Brillouin zone analysis
python3 round2_analysis.py        # KS tests + constant discrimination
```

### Output

All results go to `results/`:
- `summary.json` — timing + spectral analysis summary
- `kurtosis_sweep.json` — parameter sweep data
- `round2_results.json` — full test battery results
- `phi_analysis.json` — geometric constant analysis
- `report.html` — visual report with charts
- `charts/` — PNG charts for all analyses
- `*.csv` — raw data for external analysis

### Extending

To test with different parameters, edit the constants at the top of each Go file:
- `LogN` — ring degree (8=256, 10=1024, 12=4096)
- `Sigma` / `Bound` — Gaussian distribution parameters
- `Moduli` — NTT-friendly primes (must satisfy q ≡ 1 mod 2N)
- `NumTimingTrials` / `Trials` — number of measurements

### Requirements

- Go 1.25+ (tested with 1.25.6)
- Python 3.10+ with: numpy, scipy, matplotlib
- ~500MB disk for repos + results
- ~2 minutes for full test battery
