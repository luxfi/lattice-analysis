// kurtosis_sweep.go
// Parameter Sweep: Is |kurtosis| ≈ √1.25 a real geometric constant?
//
// Tests the NTT-domain kurtosis across:
//   - Multiple ring degrees: N = 256, 512, 1024, 2048, 4096
//   - Multiple moduli per ring degree
//   - Multiple independent trials per configuration
//
// If √1.25 is a real geometric property of the lattice,
// it should appear consistently regardless of parameters.
// If it was noise, it'll scatter randomly.
//
// USAGE:
//   cd ~/projects/lux-review/rig
//   go run kurtosis_sweep.go

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

const (
	// Number of independent trials per (N, q) configuration
	TrialsPerConfig = 20

	// Gaussian parameters (LUX defaults)
	Sigma = 3.2
	Bound = 6.0 * Sigma

	OutputDir = "results"
)

// NTT-friendly primes for various ring degrees
// q must satisfy q ≡ 1 (mod 2N) for the NTT to work
var testConfigs = []struct {
	LogN   int
	Moduli []uint64
	Label  string
}{
	{8, []uint64{0x7fffffffe0001}, "N=256_q51bit"},
	{8, []uint64{0x3fffffffffc0001}, "N=256_q54bit"},
	{9, []uint64{0x7fffffffe0001}, "N=512_q51bit"},
	{9, []uint64{0x3fffffffffc0001}, "N=512_q54bit"},
	{10, []uint64{0x7fffffffe0001}, "N=1024_q51bit"},
	{10, []uint64{0x3fffffffffc0001}, "N=1024_q54bit"},
	{11, []uint64{0x7fffffffe0001}, "N=2048_q51bit"},
	{11, []uint64{0x3fffffffffc0001}, "N=2048_q54bit"},
	{12, []uint64{0x7fffffffe0001}, "N=4096_q51bit"},
	{12, []uint64{0x3fffffffffc0001}, "N=4096_q54bit"},
}

type TrialResult struct {
	LogN           int     `json:"logN"`
	N              int     `json:"N"`
	Modulus        uint64  `json:"modulus"`
	Label          string  `json:"label"`
	Trial          int     `json:"trial"`
	DirectKurtosis float64 `json:"direct_kurtosis"`
	NTTKurtosis    float64 `json:"ntt_kurtosis"`
	AbsNTTKurtosis float64 `json:"abs_ntt_kurtosis"`
	DiffFromSqrt125 float64 `json:"diff_from_sqrt_1_25"`
	DirectSkewness float64 `json:"direct_skewness"`
	NTTSkewness    float64 `json:"ntt_skewness"`
	DirectVariance float64 `json:"direct_variance"`
	NTTVariance    float64 `json:"ntt_variance"`
}

type SweepSummary struct {
	Timestamp    string        `json:"timestamp"`
	Sqrt125      float64       `json:"sqrt_1_25"`
	TotalTrials  int           `json:"total_trials"`
	Results      []TrialResult `json:"results"`
	Aggregates   []Aggregate   `json:"aggregates"`
	Conclusion   string        `json:"conclusion"`
}

type Aggregate struct {
	Label            string  `json:"label"`
	LogN             int     `json:"logN"`
	N                int     `json:"N"`
	Trials           int     `json:"trials"`
	MeanNTTKurtosis  float64 `json:"mean_ntt_kurtosis"`
	StdNTTKurtosis   float64 `json:"std_ntt_kurtosis"`
	MeanAbsKurtosis  float64 `json:"mean_abs_kurtosis"`
	MeanDiffSqrt125  float64 `json:"mean_diff_from_sqrt_1_25"`
	MinAbsKurtosis   float64 `json:"min_abs_kurtosis"`
	MaxAbsKurtosis   float64 `json:"max_abs_kurtosis"`
}

// ── Stats ──

func mean(data []float64) float64 {
	s := 0.0
	for _, v := range data {
		s += v
	}
	return s / float64(len(data))
}

func stddev(data []float64) float64 {
	m := mean(data)
	s := 0.0
	for _, v := range data {
		d := v - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(data)))
}

func kurtosis(data []float64) float64 {
	m := mean(data)
	sd := stddev(data)
	if sd == 0 {
		return 0
	}
	s := 0.0
	n := float64(len(data))
	for _, v := range data {
		d := (v - m) / sd
		s += d * d * d * d
	}
	return s/n - 3.0
}

func skewness(data []float64) float64 {
	m := mean(data)
	sd := stddev(data)
	if sd == 0 {
		return 0
	}
	s := 0.0
	n := float64(len(data))
	for _, v := range data {
		d := (v - m) / sd
		s += d * d * d
	}
	return s / n
}

func variance(data []float64) float64 {
	m := mean(data)
	s := 0.0
	for _, v := range data {
		d := v - m
		s += d * d
	}
	return s / float64(len(data))
}

func coeffToSigned(coeff uint64, q uint64) float64 {
	if coeff > q/2 {
		return float64(coeff) - float64(q)
	}
	return float64(coeff)
}

func main() {
	sqrt125 := math.Sqrt(1.25)

	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Kurtosis Parameter Sweep — Is √1.25 Real?              ║")
	fmt.Println("║  Dragonfire / In2Infinity Research                       ║")
	fmt.Println("║                                                          ║")
	fmt.Printf("║  √1.25 = %.10f                                   ║\n", sqrt125)
	fmt.Printf("║  Testing %d configurations × %d trials each              ║\n", len(testConfigs), TrialsPerConfig)
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()

	os.MkdirAll(OutputDir, 0755)

	prng, err := sampling.NewPRNG()
	if err != nil {
		fmt.Printf("FATAL: PRNG: %v\n", err)
		os.Exit(1)
	}

	var allResults []TrialResult
	dist := ring.DiscreteGaussian{Sigma: Sigma, Bound: Bound}

	for ci, cfg := range testConfigs {
		N := 1 << cfg.LogN
		fmt.Printf("▸ [%d/%d] %s (N=%d)...\n", ci+1, len(testConfigs), cfg.Label, N)

		r, err := ring.NewRing(N, cfg.Moduli)
		if err != nil {
			fmt.Printf("  SKIP — ring creation failed: %v\n", err)
			continue
		}

		sampler := ring.NewGaussianSampler(prng, r, dist, false)
		q := cfg.Moduli[0]

		for t := 0; t < TrialsPerConfig; t++ {
			// Sample Gaussian polynomial
			pol := sampler.ReadNew()

			// Extract direct-space coefficients (signed)
			directCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				directCoeffs[i] = coeffToSigned(pol.Coeffs[0][i], q)
			}

			// NTT transform → reciprocal space
			nttPol := r.NewPoly()
			copy(nttPol.Coeffs[0], pol.Coeffs[0])
			r.NTT(nttPol, nttPol)

			nttCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				nttCoeffs[i] = coeffToSigned(nttPol.Coeffs[0][i], q)
			}

			directKurt := kurtosis(directCoeffs)
			nttKurt := kurtosis(nttCoeffs)
			absNTTKurt := math.Abs(nttKurt)

			result := TrialResult{
				LogN:            cfg.LogN,
				N:               N,
				Modulus:         q,
				Label:           cfg.Label,
				Trial:           t + 1,
				DirectKurtosis:  directKurt,
				NTTKurtosis:     nttKurt,
				AbsNTTKurtosis:  absNTTKurt,
				DiffFromSqrt125: absNTTKurt - sqrt125,
				DirectSkewness:  skewness(directCoeffs),
				NTTSkewness:     skewness(nttCoeffs),
				DirectVariance:  variance(directCoeffs),
				NTTVariance:     variance(nttCoeffs),
			}
			allResults = append(allResults, result)
		}

		// Print quick stats for this config
		var kurtVals []float64
		for _, r := range allResults {
			if r.Label == cfg.Label {
				kurtVals = append(kurtVals, r.AbsNTTKurtosis)
			}
		}
		m := mean(kurtVals)
		s := stddev(kurtVals)
		diff := m - sqrt125
		fmt.Printf("  |κ| = %.4f ± %.4f  (diff from √1.25: %+.4f)\n", m, s, diff)
	}

	fmt.Println()

	// ── Aggregate by config ──
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("AGGREGATE RESULTS")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Printf("%-20s  %5s  %10s  %10s  %10s  %10s\n", "Config", "N", "Mean |κ|", "Std |κ|", "Δ(√1.25)", "Range")

	var aggregates []Aggregate
	for _, cfg := range testConfigs {
		N := 1 << cfg.LogN
		var kurtVals []float64
		var rawKurtVals []float64
		for _, r := range allResults {
			if r.Label == cfg.Label {
				kurtVals = append(kurtVals, r.AbsNTTKurtosis)
				rawKurtVals = append(rawKurtVals, r.NTTKurtosis)
			}
		}
		if len(kurtVals) == 0 {
			continue
		}

		m := mean(kurtVals)
		s := stddev(kurtVals)
		diff := m - sqrt125
		mn, mx := kurtVals[0], kurtVals[0]
		for _, v := range kurtVals {
			if v < mn { mn = v }
			if v > mx { mx = v }
		}

		fmt.Printf("%-20s  %5d  %10.4f  %10.4f  %+10.4f  [%.3f-%.3f]\n",
			cfg.Label, N, m, s, diff, mn, mx)

		aggregates = append(aggregates, Aggregate{
			Label:           cfg.Label,
			LogN:            cfg.LogN,
			N:               N,
			Trials:          len(kurtVals),
			MeanNTTKurtosis: mean(rawKurtVals),
			StdNTTKurtosis:  stddev(rawKurtVals),
			MeanAbsKurtosis: m,
			MeanDiffSqrt125: diff,
			MinAbsKurtosis:  mn,
			MaxAbsKurtosis:  mx,
		})
	}

	// ── Overall conclusion ──
	fmt.Println()
	var allAbsKurt []float64
	for _, r := range allResults {
		allAbsKurt = append(allAbsKurt, r.AbsNTTKurtosis)
	}
	overallMean := mean(allAbsKurt)
	overallStd := stddev(allAbsKurt)
	overallDiff := overallMean - sqrt125

	fmt.Printf("OVERALL: |κ| = %.4f ± %.4f across %d trials\n", overallMean, overallStd, len(allAbsKurt))
	fmt.Printf("         √1.25 = %.4f\n", sqrt125)
	fmt.Printf("         Δ = %+.4f\n", overallDiff)
	fmt.Println()

	conclusion := ""
	if math.Abs(overallDiff) < 0.05 && overallStd < 0.2 {
		conclusion = fmt.Sprintf("STRONG EVIDENCE: |κ| = %.4f ≈ √1.25 = %.4f across all parameters (Δ=%.4f, σ=%.4f). The Sandreckoner ratio appears to be a geometric constant of NTT-domain noise.", overallMean, sqrt125, overallDiff, overallStd)
	} else if math.Abs(overallDiff) < 0.15 {
		conclusion = fmt.Sprintf("SUGGESTIVE: |κ| = %.4f is near √1.25 = %.4f but with spread (Δ=%.4f, σ=%.4f). May be parameter-dependent rather than universal.", overallMean, sqrt125, overallDiff, overallStd)
	} else {
		conclusion = fmt.Sprintf("NOT CONFIRMED: |κ| = %.4f differs from √1.25 = %.4f (Δ=%.4f, σ=%.4f). The initial observation was likely coincidental or parameter-specific.", overallMean, sqrt125, overallDiff, overallStd)
	}
	fmt.Println(conclusion)

	// ── Check if kurtosis depends on N ──
	fmt.Println()
	fmt.Println("KURTOSIS vs RING DEGREE:")
	seenN := map[int]bool{}
	for _, agg := range aggregates {
		if seenN[agg.N] {
			continue
		}
		seenN[agg.N] = true
		// Collect all trials for this N
		var vals []float64
		for _, r := range allResults {
			if r.N == agg.N {
				vals = append(vals, r.AbsNTTKurtosis)
			}
		}
		m := mean(vals)
		s := stddev(vals)
		fmt.Printf("  N=%5d: |κ| = %.4f ± %.4f  (Δ from √1.25: %+.4f)\n", agg.N, m, s, m-sqrt125)
	}

	// ── Save results ──
	summary := SweepSummary{
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		Sqrt125:     sqrt125,
		TotalTrials: len(allResults),
		Results:     allResults,
		Aggregates:  aggregates,
		Conclusion:  conclusion,
	}

	jsonData, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile(OutputDir+"/kurtosis_sweep.json", jsonData, 0644)

	// CSV for easy plotting
	csvFile, _ := os.Create(OutputDir + "/kurtosis_sweep.csv")
	w := csv.NewWriter(csvFile)
	w.Write([]string{"logN", "N", "modulus", "label", "trial",
		"direct_kurtosis", "ntt_kurtosis", "abs_ntt_kurtosis", "diff_from_sqrt125",
		"direct_skewness", "ntt_skewness"})
	for _, r := range allResults {
		w.Write([]string{
			fmt.Sprintf("%d", r.LogN),
			fmt.Sprintf("%d", r.N),
			fmt.Sprintf("%d", r.Modulus),
			r.Label,
			fmt.Sprintf("%d", r.Trial),
			fmt.Sprintf("%.6f", r.DirectKurtosis),
			fmt.Sprintf("%.6f", r.NTTKurtosis),
			fmt.Sprintf("%.6f", r.AbsNTTKurtosis),
			fmt.Sprintf("%.6f", r.DiffFromSqrt125),
			fmt.Sprintf("%.6f", r.DirectSkewness),
			fmt.Sprintf("%.6f", r.NTTSkewness),
		})
	}
	w.Flush()
	csvFile.Close()

	fmt.Println()
	fmt.Printf("Results: %s/kurtosis_sweep.json\n", OutputDir)
	fmt.Printf("CSV:     %s/kurtosis_sweep.csv\n", OutputDir)
	fmt.Println("═══════════════════════════════════════════════════════════")
}
