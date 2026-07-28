// lattice_analysis.go
// Reciprocal-Space Security Analysis Rig for LUX Lattice Library
//
// Instruments the REAL luxfi/lattice library to:
// 1. Capture Gaussian noise samples from the actual sampler
// 2. Analyse distribution in both direct and NTT (reciprocal) space
// 3. Measure timing of critical crypto operations for side-channel detection
// 4. Output data for spectral analysis (Python)
//
// USAGE:
//   cd ~/projects/lux-review/rig
//   go run lattice_analysis.go
//
// OUTPUT:
//   results/noise_samples_direct.csv
//   results/noise_samples_ntt.csv
//   results/timing_ntt.csv
//   results/timing_gaussian.csv
//   results/summary.json

package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/luxfi/lattice/v7/ring"
	"github.com/luxfi/lattice/v7/utils/sampling"
)

// ============================================================
// CONFIGURATION
// ============================================================

const (
	// Ring degree as log2 — 2^LogN coefficients
	// Common values: 10 (1024), 11 (2048), 12 (4096)
	LogN = 10

	// Number of noise samples to collect (multiple polys if needed)
	NumSamples = 10000

	// Number of timing measurements per operation
	NumTimingTrials = 5000

	// Gaussian parameters — matching LUX defaults
	Sigma = 3.2
	Bound = 6.0 * Sigma // 19.2

	OutputDir = "results"
)

// NTT-friendly prime for ring degree 2^10 = 1024
// q ≡ 1 (mod 2N) required for NTT
var Moduli = []uint64{0x7fffffffe0001} // 51-bit NTT-friendly prime

// ============================================================
// DATA STRUCTURES
// ============================================================

type TimingResult struct {
	Operation string  `json:"operation"`
	Trials    int     `json:"trials"`
	Mean      float64 `json:"mean_ns"`
	StdDev    float64 `json:"stddev_ns"`
	Min       float64 `json:"min_ns"`
	Max       float64 `json:"max_ns"`
	CV        float64 `json:"coefficient_of_variation"`
	Verdict   string  `json:"verdict"`
}

type SpectralResult struct {
	Domain       string  `json:"domain"`
	Mean         float64 `json:"mean"`
	Variance     float64 `json:"variance"`
	Skewness     float64 `json:"skewness"`
	Kurtosis     float64 `json:"kurtosis"`
	MaxFreqPower float64 `json:"max_frequency_power,omitempty"`
	Verdict      string  `json:"verdict"`
}

type AnalysisSummary struct {
	Timestamp       string           `json:"timestamp"`
	RingDegree      int              `json:"ring_degree"`
	NumSamples      int              `json:"num_samples"`
	Sigma           float64          `json:"sigma"`
	Bound           float64          `json:"bound"`
	Moduli          []uint64         `json:"moduli"`
	TimingResults   []TimingResult   `json:"timing_analysis"`
	SpectralResults []SpectralResult `json:"spectral_analysis"`
	Findings        []string         `json:"findings"`
}

// ============================================================
// STATISTICAL UTILITIES
// ============================================================

func mean(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func variance(data []float64) float64 {
	m := mean(data)
	sum := 0.0
	for _, v := range data {
		d := v - m
		sum += d * d
	}
	return sum / float64(len(data))
}

func stddev(data []float64) float64 {
	return math.Sqrt(variance(data))
}

func skewness(data []float64) float64 {
	m := mean(data)
	sd := stddev(data)
	if sd == 0 {
		return 0
	}
	sum := 0.0
	n := float64(len(data))
	for _, v := range data {
		d := (v - m) / sd
		sum += d * d * d
	}
	return sum / n
}

func kurtosis(data []float64) float64 {
	m := mean(data)
	sd := stddev(data)
	if sd == 0 {
		return 0
	}
	sum := 0.0
	n := float64(len(data))
	for _, v := range data {
		d := (v - m) / sd
		sum += d * d * d * d
	}
	return sum/n - 3.0 // Excess kurtosis (normal = 0)
}

func minMax(data []float64) (float64, float64) {
	mn, mx := data[0], data[0]
	for _, v := range data[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

// ============================================================
// TIMING ANALYSIS
// ============================================================

func measureOperation(name string, trials int, fn func()) TimingResult {
	timings := make([]float64, trials)

	// Warm up
	for i := 0; i < 100; i++ {
		fn()
	}

	// Measure
	for i := 0; i < trials; i++ {
		start := time.Now()
		fn()
		timings[i] = float64(time.Since(start).Nanoseconds())
	}

	m := mean(timings)
	sd := stddev(timings)
	mn, mx := minMax(timings)
	cv := sd / m

	verdict := "OK — timing appears constant"
	if cv > 0.1 {
		verdict = fmt.Sprintf("WARNING: High timing variance (CV=%.3f) — potential side-channel", cv)
	}
	if cv > 0.3 {
		verdict = fmt.Sprintf("CRITICAL: Very high timing variance (CV=%.3f) — likely non-constant-time", cv)
	}

	return TimingResult{
		Operation: name,
		Trials:    trials,
		Mean:      m,
		StdDev:    sd,
		Min:       mn,
		Max:       mx,
		CV:        cv,
		Verdict:   verdict,
	}
}

// ============================================================
// DISTRIBUTION ANALYSIS
// ============================================================

func analyseDistribution(samples []float64, domain string) SpectralResult {
	m := mean(samples)
	v := variance(samples)
	sk := skewness(samples)
	ku := kurtosis(samples)

	verdict := "OK"
	if math.Abs(sk) > 0.1 {
		verdict = fmt.Sprintf("WARNING: Non-zero skewness (%.4f) — distribution asymmetry", sk)
	}
	if math.Abs(ku) > 0.5 {
		verdict = fmt.Sprintf("WARNING: Non-Gaussian kurtosis (%.4f) — heavy/light tails", ku)
	}
	if math.Abs(sk) > 0.1 && math.Abs(ku) > 0.5 {
		verdict = fmt.Sprintf("CRITICAL: Both skewness (%.4f) and kurtosis (%.4f) anomalous — noise distribution compromised", sk, ku)
	}

	return SpectralResult{
		Domain:   domain,
		Mean:     m,
		Variance: v,
		Skewness: sk,
		Kurtosis: ku,
		Verdict:  verdict,
	}
}

// ============================================================
// CSV OUTPUT
// ============================================================

func writeSamplesCSV(filename string, samples []float64) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"index", "value"})
	for i, s := range samples {
		w.Write([]string{fmt.Sprintf("%d", i), fmt.Sprintf("%.10f", s)})
	}
	w.Flush()
	return w.Error()
}

func writeTimingsCSV(filename string, timings []float64) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"trial", "nanoseconds"})
	for i, t := range timings {
		w.Write([]string{fmt.Sprintf("%d", i), fmt.Sprintf("%.0f", t)})
	}
	w.Flush()
	return w.Error()
}

// ============================================================
// COEFFICIENT EXTRACTION UTILITIES
// ============================================================

// coeffToSigned converts a coefficient in [0, q) to signed representation
// centered around 0: if coeff > q/2, return coeff - q
func coeffToSigned(coeff uint64, q uint64) float64 {
	if coeff > q/2 {
		return float64(coeff) - float64(q)
	}
	return float64(coeff)
}

// extractSignedCoeffs extracts the first n coefficients from a polynomial
// as signed float64 values centered around 0
func extractSignedCoeffs(pol ring.Poly, q uint64, n int) []float64 {
	coeffs := pol.Coeffs[0] // First RNS component (level 0)
	result := make([]float64, n)
	for i := 0; i < n && i < len(coeffs); i++ {
		result[i] = coeffToSigned(coeffs[i], q)
	}
	return result
}

// ============================================================
// MAIN
// ============================================================

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║  LUX Lattice — Reciprocal-Space Security Analysis       ║")
	fmt.Println("║  Dragonfire Research Rig v0.2 (LIVE — real lattice lib)  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()

	os.MkdirAll(OutputDir, 0755)

	// ─── Initialize Ring ─────────────────────────────────────────
	fmt.Println("▸ Initializing ring parameters...")
	N := 1 << LogN
	fmt.Printf("  Ring degree N = %d (LogN = %d)\n", N, LogN)
	fmt.Printf("  Modulus q = %d (0x%x)\n", Moduli[0], Moduli[0])
	fmt.Printf("  Gaussian σ = %.1f, bound = %.1f\n", Sigma, Bound)

	r, err := ring.NewRing(N, Moduli)
	if err != nil {
		fmt.Printf("  ERROR creating ring: %v\n", err)
		fmt.Println("  Trying alternative modulus...")
		// Fallback: use a known good NTT-friendly prime for N=1024
		// q must satisfy q ≡ 1 (mod 2N)
		Moduli[0] = 0x3fffffffffc0001 // 54-bit prime
		r, err = ring.NewRing(N, Moduli)
		if err != nil {
			fmt.Printf("  FATAL: Cannot create ring: %v\n", err)
			os.Exit(1)
		}
	}
	fmt.Printf("  Ring created successfully: N=%d, levels=%d\n\n", r.N(), r.Level()+1)

	// ─── Initialize PRNG ─────────────────────────────────────────
	prng, err := sampling.NewPRNG()
	if err != nil {
		fmt.Printf("FATAL: Cannot create PRNG: %v\n", err)
		os.Exit(1)
	}

	// ─── Initialize Gaussian Sampler ─────────────────────────────
	dist := ring.DiscreteGaussian{Sigma: Sigma, Bound: Bound}
	gaussianSampler := ring.NewGaussianSampler(prng, r, dist, false)

	summary := AnalysisSummary{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		RingDegree: N,
		NumSamples: NumSamples,
		Sigma:      Sigma,
		Bound:      Bound,
		Moduli:     Moduli,
	}

	q := Moduli[0]

	// ─── Step 1: Gaussian Noise Sampling (Direct Space) ──────────
	fmt.Println("▸ Step 1: Collecting Gaussian noise samples from REAL sampler...")

	numPolys := (NumSamples + N - 1) / N // How many polys to fill NumSamples
	directSamples := make([]float64, 0, NumSamples)

	for p := 0; p < numPolys; p++ {
		pol := gaussianSampler.ReadNew()
		coeffs := extractSignedCoeffs(pol, q, N)
		directSamples = append(directSamples, coeffs...)
	}
	directSamples = directSamples[:NumSamples] // Trim to exact count

	writeSamplesCSV(OutputDir+"/noise_samples_direct.csv", directSamples)
	fmt.Printf("  Collected %d REAL Gaussian samples → %s/noise_samples_direct.csv\n\n", NumSamples, OutputDir)

	// ─── Step 2: NTT Transform (Direct → Reciprocal Space) ──────
	fmt.Println("▸ Step 2: Applying REAL NTT transform (direct → reciprocal space)...")

	// Sample a fresh polynomial and transform it
	nttPol := gaussianSampler.ReadNew()
	nttOut := r.NewPoly()

	// Copy before transform (NTT is in-place capable)
	copy(nttOut.Coeffs[0], nttPol.Coeffs[0])

	// Forward NTT — moves to reciprocal space (frequency domain)
	r.NTT(nttOut, nttOut)

	reciprocalSamples := extractSignedCoeffs(nttOut, q, N)

	writeSamplesCSV(OutputDir+"/noise_samples_ntt.csv", reciprocalSamples)
	fmt.Printf("  Transformed %d coefficients via NTT → %s/noise_samples_ntt.csv\n\n", N, OutputDir)

	// ─── Step 3: Statistical Analysis ────────────────────────────
	fmt.Println("▸ Step 3: Analysing distributions...")

	directResult := analyseDistribution(directSamples, "direct_space")
	fmt.Printf("  Direct space:     mean=%.4f var=%.4f skew=%.4f kurt=%.4f [%s]\n",
		directResult.Mean, directResult.Variance, directResult.Skewness, directResult.Kurtosis, directResult.Verdict)

	reciprocalResult := analyseDistribution(reciprocalSamples, "reciprocal_space")
	fmt.Printf("  Reciprocal space: mean=%.4f var=%.4f skew=%.4f kurt=%.4f [%s]\n",
		reciprocalResult.Mean, reciprocalResult.Variance, reciprocalResult.Skewness, reciprocalResult.Kurtosis, reciprocalResult.Verdict)

	summary.SpectralResults = append(summary.SpectralResults, directResult, reciprocalResult)
	fmt.Println()

	// ─── Step 4: Timing Analysis (Side-Channel Detection) ────────
	fmt.Println("▸ Step 4: Timing analysis on REAL lattice operations...")

	// 4a: Gaussian sampling timing
	fmt.Println("  Measuring Gaussian sampler timing...")
	gaussTimings := make([]float64, NumTimingTrials)
	// Warm up
	for i := 0; i < 50; i++ {
		gaussianSampler.ReadNew()
	}
	for i := 0; i < NumTimingTrials; i++ {
		start := time.Now()
		gaussianSampler.ReadNew()
		gaussTimings[i] = float64(time.Since(start).Nanoseconds())
	}
	gaussTiming := TimingResult{
		Operation: "Gaussian_Sample_Real",
		Trials:    NumTimingTrials,
		Mean:      mean(gaussTimings),
		StdDev:    stddev(gaussTimings),
		CV:        stddev(gaussTimings) / mean(gaussTimings),
	}
	gaussTiming.Min, gaussTiming.Max = minMax(gaussTimings)
	gaussTiming.Verdict = "OK — timing appears constant"
	if gaussTiming.CV > 0.1 {
		gaussTiming.Verdict = fmt.Sprintf("WARNING: High timing variance (CV=%.3f) — potential side-channel", gaussTiming.CV)
	}
	if gaussTiming.CV > 0.3 {
		gaussTiming.Verdict = fmt.Sprintf("CRITICAL: Very high timing variance (CV=%.3f) — likely non-constant-time", gaussTiming.CV)
	}
	writeTimingsCSV(OutputDir+"/timing_gaussian.csv", gaussTimings)
	fmt.Printf("  Gaussian: mean=%.0fns stddev=%.0fns CV=%.3f [%s]\n",
		gaussTiming.Mean, gaussTiming.StdDev, gaussTiming.CV, gaussTiming.Verdict)

	// 4b: NTT timing
	fmt.Println("  Measuring NTT transform timing...")
	nttTimings := make([]float64, NumTimingTrials)
	testPol := gaussianSampler.ReadNew()
	outPol := r.NewPoly()
	// Warm up
	for i := 0; i < 50; i++ {
		r.NTT(testPol, outPol)
	}
	for i := 0; i < NumTimingTrials; i++ {
		start := time.Now()
		r.NTT(testPol, outPol)
		nttTimings[i] = float64(time.Since(start).Nanoseconds())
	}
	nttTiming := TimingResult{
		Operation: "NTT_Forward_Real",
		Trials:    NumTimingTrials,
		Mean:      mean(nttTimings),
		StdDev:    stddev(nttTimings),
		CV:        stddev(nttTimings) / mean(nttTimings),
	}
	nttTiming.Min, nttTiming.Max = minMax(nttTimings)
	nttTiming.Verdict = "OK — timing appears constant"
	if nttTiming.CV > 0.1 {
		nttTiming.Verdict = fmt.Sprintf("WARNING: High timing variance (CV=%.3f) — potential side-channel", nttTiming.CV)
	}
	if nttTiming.CV > 0.3 {
		nttTiming.Verdict = fmt.Sprintf("CRITICAL: Very high timing variance (CV=%.3f) — likely non-constant-time", nttTiming.CV)
	}
	writeTimingsCSV(OutputDir+"/timing_ntt.csv", nttTimings)
	fmt.Printf("  NTT: mean=%.0fns stddev=%.0fns CV=%.3f [%s]\n",
		nttTiming.Mean, nttTiming.StdDev, nttTiming.CV, nttTiming.Verdict)

	// 4c: INTT timing
	fmt.Println("  Measuring INTT (inverse NTT) timing...")
	inttTimings := make([]float64, NumTimingTrials)
	nttedPol := r.NewPoly()
	r.NTT(testPol, nttedPol) // Pre-transform for INTT input
	inttOut := r.NewPoly()
	for i := 0; i < 50; i++ {
		r.INTT(nttedPol, inttOut)
	}
	for i := 0; i < NumTimingTrials; i++ {
		start := time.Now()
		r.INTT(nttedPol, inttOut)
		inttTimings[i] = float64(time.Since(start).Nanoseconds())
	}
	inttTiming := TimingResult{
		Operation: "INTT_Backward_Real",
		Trials:    NumTimingTrials,
		Mean:      mean(inttTimings),
		StdDev:    stddev(inttTimings),
		CV:        stddev(inttTimings) / mean(inttTimings),
	}
	inttTiming.Min, inttTiming.Max = minMax(inttTimings)
	inttTiming.Verdict = "OK — timing appears constant"
	if inttTiming.CV > 0.1 {
		inttTiming.Verdict = fmt.Sprintf("WARNING: High timing variance (CV=%.3f) — potential side-channel", inttTiming.CV)
	}
	writeTimingsCSV(OutputDir+"/timing_intt.csv", inttTimings)
	fmt.Printf("  INTT: mean=%.0fns stddev=%.0fns CV=%.3f [%s]\n",
		inttTiming.Mean, inttTiming.StdDev, inttTiming.CV, inttTiming.Verdict)

	// 4d: Input-dependent timing test — the real side-channel check
	// Sample multiple different polynomials and see if timing varies with input
	fmt.Println("  Measuring input-dependent Gaussian timing (side-channel test)...")
	lowNormTimings := make([]float64, 0, NumTimingTrials/2)
	highNormTimings := make([]float64, 0, NumTimingTrials/2)

	for i := 0; i < NumTimingTrials; i++ {
		start := time.Now()
		pol := gaussianSampler.ReadNew()
		elapsed := float64(time.Since(start).Nanoseconds())

		// Compute L2 norm of the sampled polynomial
		norm := 0.0
		for _, c := range pol.Coeffs[0] {
			v := coeffToSigned(c, q)
			norm += v * v
		}
		norm = math.Sqrt(norm)

		// Classify by norm magnitude
		if norm < float64(N)*Sigma*0.8 {
			lowNormTimings = append(lowNormTimings, elapsed)
		} else {
			highNormTimings = append(highNormTimings, elapsed)
		}
	}

	if len(lowNormTimings) > 10 && len(highNormTimings) > 10 {
		lowMean := mean(lowNormTimings)
		highMean := mean(highNormTimings)
		timingDiff := math.Abs(highMean-lowMean) / ((highMean + lowMean) / 2)
		fmt.Printf("  Input-dependent: low-norm mean=%.0fns (%d), high-norm mean=%.0fns (%d), diff=%.4f%%\n",
			lowMean, len(lowNormTimings), highMean, len(highNormTimings), timingDiff*100)
		if timingDiff > 0.05 {
			fmt.Printf("  WARNING: >5%% timing difference between low/high norm outputs — input-dependent timing!\n")
		}
	}

	summary.TimingResults = append(summary.TimingResults, gaussTiming, nttTiming, inttTiming)
	fmt.Println()

	// ─── Step 5: Compile Findings ────────────────────────────────
	fmt.Println("▸ Step 5: Compiling findings...")

	findings := []string{}

	for _, tr := range summary.TimingResults {
		if tr.CV > 0.1 {
			findings = append(findings, fmt.Sprintf("TIMING: %s has CV=%.3f — non-constant-time risk", tr.Operation, tr.CV))
		}
	}

	for _, sr := range summary.SpectralResults {
		if math.Abs(sr.Skewness) > 0.1 {
			findings = append(findings, fmt.Sprintf("SPECTRAL: %s has skewness=%.4f — distribution asymmetry", sr.Domain, sr.Skewness))
		}
		if math.Abs(sr.Kurtosis) > 0.5 {
			findings = append(findings, fmt.Sprintf("SPECTRAL: %s has kurtosis=%.4f — non-Gaussian tails", sr.Domain, sr.Kurtosis))
		}
	}

	// Reciprocal space energy redistribution check
	directVar := directResult.Variance
	reciprocalVar := reciprocalResult.Variance
	if reciprocalVar > 0 {
		varRatio := directVar / reciprocalVar
		if varRatio > 10 || varRatio < 0.1 {
			findings = append(findings, fmt.Sprintf(
				"RECIPROCAL: Variance ratio (direct/reciprocal) = %.2f — significant energy redistribution in frequency domain",
				varRatio))
		}
	}

	// Input-dependent timing finding
	if len(lowNormTimings) > 10 && len(highNormTimings) > 10 {
		timingDiff := math.Abs(mean(highNormTimings)-mean(lowNormTimings)) / ((mean(highNormTimings) + mean(lowNormTimings)) / 2)
		if timingDiff > 0.05 {
			findings = append(findings, fmt.Sprintf(
				"SIDE-CHANNEL: Gaussian sampler timing correlates with output norm (%.2f%% difference) — rejection sampling leakage",
				timingDiff*100))
		}
	}

	if len(findings) == 0 {
		findings = append(findings, "No anomalies detected in this run. Consider increasing sample size or testing at different ring degrees.")
	}

	summary.Findings = findings

	// ─── Output Summary ──────────────────────────────────────────
	summaryJSON, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile(OutputDir+"/summary.json", summaryJSON, 0644)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("FINDINGS:")
	for _, f := range findings {
		fmt.Println("  •", f)
	}
	fmt.Println()
	fmt.Printf("Results written to %s/\n", OutputDir)
	fmt.Println("Run: python3 spectral_analysis.py for visualisation.")
	fmt.Println("═══════════════════════════════════════════════════════════")
}
