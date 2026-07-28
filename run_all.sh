#!/bin/bash
# LUX Lattice Security Analysis — Full Pipeline
# Dragonfire Research Rig
#
# Runs all Go harnesses + Python analysis in order.
# Produces results/ directory with JSON, CSV, charts, and HTML report.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  LUX Lattice Security Analysis — Full Pipeline           ║"
echo "║  Dragonfire Research Rig                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Prerequisites ──
echo "▸ Checking prerequisites..."
if ! command -v go &> /dev/null; then
    echo "  ERROR: Go not found. Install Go 1.25+"
    exit 1
fi
echo "  Go $(go version | awk '{print $3}')"

if ! command -v python3 &> /dev/null; then
    echo "  ERROR: Python3 not found."
    exit 1
fi
echo "  Python $(python3 --version | awk '{print $2}')"

# Check Go module is set up
if [ ! -f "go.mod" ]; then
    echo "  Setting up Go module..."
    go mod init dragonfire/lattice-analysis
    go mod edit -replace github.com/luxfi/lattice/v7=../lattice
    go mod tidy
fi
echo ""

mkdir -p results

# ── Stage 1: Timing + Noise Analysis ──
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 1: Timing & Noise Analysis"
echo "════════════════════════════════════════════════════════════"
go run lattice_analysis.go
echo ""

# ── Stage 2: Kurtosis Parameter Sweep ──
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 2: Kurtosis Parameter Sweep"
echo "════════════════════════════════════════════════════════════"
go run kurtosis_sweep.go
echo ""

# ── Stage 3: Round 2 Test Battery ──
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 3: Round 2 Test Battery (A-E)"
echo "════════════════════════════════════════════════════════════"
go run round2_tests.go
echo ""

# ── Stage 4: Python Spectral Analysis ──
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 4: Python Analysis"
echo "════════════════════════════════════════════════════════════"

python3 -c "import numpy" 2>/dev/null || pip3 install --user numpy --quiet
python3 -c "import scipy" 2>/dev/null || pip3 install --user scipy --quiet
python3 -c "import matplotlib" 2>/dev/null || pip3 install --user matplotlib --quiet

echo "  Running spectral analysis..."
python3 spectral_analysis.py

echo "  Running phi-geometric analysis..."
python3 phi_analysis.py

echo "  Running round 2 analysis (KS tests, constant discrimination)..."
python3 round2_analysis.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Results:  $(pwd)/results/"
echo "  Report:   $(pwd)/results/report.html"
echo "  Charts:   $(pwd)/results/charts/"
echo ""
echo "  Key files:"
echo "    summary.json         — timing + spectral findings"
echo "    kurtosis_sweep.json  — parameter sweep (N, sigma, q)"
echo "    round2_results.json  — full test battery"
echo "    phi_analysis.json    — geometric constant analysis"
echo ""
echo "  Open results/report.html in a browser for visual report."
echo "════════════════════════════════════════════════════════════"
