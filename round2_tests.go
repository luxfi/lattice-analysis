// round2_tests.go
// Round 2: WHY is NTT kurtosis ≈ -1.200? Is it exploitable?
//
// Test A: KS-ready data (normalised NTT coefficients for Python)
// Test B: Vary Gaussian sigma (1.0, 2.0, 3.2, 5.0, 8.0, 16.0)
// Test C: Unbounded vs bounded sampling
// Test D: Vary modulus size
// Test E: Plain DFT comparison
//
// USAGE:
//   go run round2_tests.go
//   python3 round2_analysis.py

package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"time"

	"github.com/luxfi/lattice/v7/ring"
	"github.com/luxfi/lattice/v7/utils/sampling"
)

const (
	Trials    = 20
	OutputDir = "results"
)

// ── Stats ──

func mean(d []float64) float64 {
	s := 0.0
	for _, v := range d {
		s += v
	}
	return s / float64(len(d))
}

func stddev(d []float64) float64 {
	m := mean(d)
	s := 0.0
	for _, v := range d {
		x := v - m
		s += x * x
	}
	return math.Sqrt(s / float64(len(d)))
}

func kurtosis(d []float64) float64 {
	m := mean(d)
	sd := stddev(d)
	if sd == 0 {
		return 0
	}
	s := 0.0
	n := float64(len(d))
	for _, v := range d {
		x := (v - m) / sd
		s += x * x * x * x
	}
	return s/n - 3.0
}

func coeffToSigned(c uint64, q uint64) float64 {
	if c > q/2 {
		return float64(c) - float64(q)
	}
	return float64(c)
}

// ── Result types ──

type TestResult struct {
	Test      string  `json:"test"`
	Label     string  `json:"label"`
	N         int     `json:"N"`
	Sigma     float64 `json:"sigma,omitempty"`
	LogQ      int     `json:"log_q,omitempty"`
	Modulus   uint64  `json:"modulus,omitempty"`
	Trial     int     `json:"trial"`
	DirectK   float64 `json:"direct_kurtosis"`
	NTTK      float64 `json:"ntt_kurtosis"`
	AbsNTTK   float64 `json:"abs_ntt_kurtosis"`
}

type AggResult struct {
	Test     string  `json:"test"`
	Label    string  `json:"label"`
	N        int     `json:"N"`
	Param    string  `json:"parameter"`
	MeanK    float64 `json:"mean_ntt_kurtosis"`
	StdK     float64 `json:"std_ntt_kurtosis"`
	MeanAbsK float64 `json:"mean_abs_kurtosis"`
	Trials   int     `json:"trials"`
}

var allResults []TestResult
var aggResults []AggResult

func aggregate(test, label string, n int, param string, results []TestResult) {
	var kvals []float64
	var absvals []float64
	for _, r := range results {
		kvals = append(kvals, r.NTTK)
		absvals = append(absvals, r.AbsNTTK)
	}
	aggResults = append(aggResults, AggResult{
		Test:     test,
		Label:    label,
		N:        n,
		Param:    param,
		MeanK:    mean(kvals),
		StdK:     stddev(kvals),
		MeanAbsK: mean(absvals),
		Trials:   len(kvals),
	})
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Round 2: WHY is NTT kurtosis ≈ -1.200?                ║")
	fmt.Println("║  Candidate constants:                                    ║")
	fmt.Printf("║    -6/5       = -1.200000  (uniform distribution)        ║\n")
	fmt.Printf("║    -√6/2      = -%.6f  (rhombic dodecahedron)       ║\n", math.Sqrt(6)/2)
	fmt.Printf("║    -√1.25     = -%.6f  (Sandreckoner)               ║\n", math.Sqrt(1.25))
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()

	os.MkdirAll(OutputDir, 0755)

	prng, _ := sampling.NewPRNG()
	baseQ := uint64(0x7fffffffe0001) // 51-bit NTT-friendly prime

	// ═══════════════════════════════════════════════════════════
	// TEST A: Generate normalised NTT data for KS test (Python)
	// ═══════════════════════════════════════════════════════════
	fmt.Println("▸ TEST A: Generating normalised NTT data for KS test...")

	for _, logN := range []int{8, 9, 10, 11, 12} {
		N := 1 << logN
		r, err := ring.NewRing(N, []uint64{baseQ})
		if err != nil {
			fmt.Printf("  SKIP N=%d: %v\n", N, err)
			continue
		}
		dist := ring.DiscreteGaussian{Sigma: 3.2, Bound: 19.2}
		sampler := ring.NewGaussianSampler(prng, r, dist, false)

		// Generate multiple polys worth for good statistics
		label := fmt.Sprintf("A_N%d", N)
		fname := fmt.Sprintf("%s/ks_ntt_N%d.csv", OutputDir, N)
		f, _ := os.Create(fname)
		w := csv.NewWriter(f)
		w.Write([]string{"raw", "normalised", "signed"})

		var trialResults []TestResult
		for t := 0; t < Trials; t++ {
			pol := sampler.ReadNew()
			directCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				directCoeffs[i] = coeffToSigned(pol.Coeffs[0][i], baseQ)
			}

			nttPol := r.NewPoly()
			copy(nttPol.Coeffs[0], pol.Coeffs[0])
			r.NTT(nttPol, nttPol)

			nttCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				raw := nttPol.Coeffs[0][i]
				nttCoeffs[i] = coeffToSigned(raw, baseQ)
				// Write normalised (0 to 1 range) for KS test
				norm := float64(raw) / float64(baseQ)
				signed := coeffToSigned(raw, baseQ)
				w.Write([]string{
					fmt.Sprintf("%d", raw),
					fmt.Sprintf("%.10f", norm),
					fmt.Sprintf("%.4f", signed),
				})
			}

			dk := kurtosis(directCoeffs)
			nk := kurtosis(nttCoeffs)
			tr := TestResult{Test: "A", Label: label, N: N, Trial: t + 1, DirectK: dk, NTTK: nk, AbsNTTK: math.Abs(nk)}
			trialResults = append(trialResults, tr)
			allResults = append(allResults, tr)
		}
		w.Flush()
		f.Close()

		aggregate("A", label, N, "KS_data", trialResults)
		fmt.Printf("  N=%d: κ=%.4f±%.4f → %s\n", N, mean(extract(trialResults)), stddev(extract(trialResults)), fname)
	}
	fmt.Println()

	// ═══════════════════════════════════════════════════════════
	// TEST B: Vary Gaussian sigma
	// ═══════════════════════════════════════════════════════════
	fmt.Println("▸ TEST B: Varying Gaussian sigma...")

	N := 1024
	r1024, _ := ring.NewRing(N, []uint64{baseQ})

	for _, sigma := range []float64{1.0, 2.0, 3.2, 5.0, 8.0, 16.0, 32.0, 64.0} {
		bound := 6.0 * sigma
		dist := ring.DiscreteGaussian{Sigma: sigma, Bound: bound}
		sampler := ring.NewGaussianSampler(prng, r1024, dist, false)
		label := fmt.Sprintf("B_sigma%.1f", sigma)

		var trialResults []TestResult
		for t := 0; t < Trials; t++ {
			pol := sampler.ReadNew()
			directCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				directCoeffs[i] = coeffToSigned(pol.Coeffs[0][i], baseQ)
			}

			nttPol := r1024.NewPoly()
			copy(nttPol.Coeffs[0], pol.Coeffs[0])
			r1024.NTT(nttPol, nttPol)

			nttCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				nttCoeffs[i] = coeffToSigned(nttPol.Coeffs[0][i], baseQ)
			}

			dk := kurtosis(directCoeffs)
			nk := kurtosis(nttCoeffs)
			tr := TestResult{Test: "B", Label: label, N: N, Sigma: sigma, Trial: t + 1, DirectK: dk, NTTK: nk, AbsNTTK: math.Abs(nk)}
			trialResults = append(trialResults, tr)
			allResults = append(allResults, tr)
		}

		aggregate("B", label, N, fmt.Sprintf("sigma=%.1f", sigma), trialResults)
		fmt.Printf("  σ=%5.1f: κ=%.4f±%.4f  direct_κ=%.4f\n",
			sigma, mean(extract(trialResults)), stddev(extract(trialResults)),
			mean(extractDirect(trialResults)))
	}
	fmt.Println()

	// ═══════════════════════════════════════════════════════════
	// TEST C: Unbounded — use huge bound to approximate no truncation
	// ═══════════════════════════════════════════════════════════
	fmt.Println("▸ TEST C: Bounded vs 'unbounded' (huge bound)...")

	for _, boundMult := range []float64{3.0, 6.0, 10.0, 20.0, 50.0, 100.0} {
		sigma := 3.2
		bound := boundMult * sigma
		dist := ring.DiscreteGaussian{Sigma: sigma, Bound: bound}
		sampler := ring.NewGaussianSampler(prng, r1024, dist, false)
		label := fmt.Sprintf("C_bound%.0fx", boundMult)

		var trialResults []TestResult
		for t := 0; t < Trials; t++ {
			pol := sampler.ReadNew()
			directCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				directCoeffs[i] = coeffToSigned(pol.Coeffs[0][i], baseQ)
			}

			nttPol := r1024.NewPoly()
			copy(nttPol.Coeffs[0], pol.Coeffs[0])
			r1024.NTT(nttPol, nttPol)

			nttCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				nttCoeffs[i] = coeffToSigned(nttPol.Coeffs[0][i], baseQ)
			}

			dk := kurtosis(directCoeffs)
			nk := kurtosis(nttCoeffs)
			tr := TestResult{Test: "C", Label: label, N: N, Sigma: sigma, Trial: t + 1, DirectK: dk, NTTK: nk, AbsNTTK: math.Abs(nk)}
			trialResults = append(trialResults, tr)
			allResults = append(allResults, tr)
		}

		aggregate("C", label, N, fmt.Sprintf("bound=%.0fσ", boundMult), trialResults)
		fmt.Printf("  bound=%3.0fσ: ntt_κ=%.4f±%.4f  direct_κ=%.4f\n",
			boundMult, mean(extract(trialResults)), stddev(extract(trialResults)),
			mean(extractDirect(trialResults)))
	}
	fmt.Println()

	// ═══════════════════════════════════════════════════════════
	// TEST D: Vary modulus size
	// ═══════════════════════════════════════════════════════════
	fmt.Println("▸ TEST D: Varying modulus size...")

	// NTT-friendly primes at different bit sizes
	// q ≡ 1 (mod 2N) where N=1024, so q ≡ 1 (mod 2048)
	testModuli := []struct {
		q     uint64
		logQ  int
		label string
	}{
		{12289, 14, "q14bit"},         // 2^13 + 2^12 + 1, classic for N=1024
		{40961, 16, "q16bit"},         // common small NTT prime
		{65537, 17, "q17bit"},         // Fermat prime
		{786433, 20, "q20bit"},        // 2^20 range
		{1073479681, 30, "q30bit"},    // ~2^30
		{0x7fffffffe0001, 51, "q51bit"}, // our standard
	}

	for _, tm := range testModuli {
		rTest, err := ring.NewRing(N, []uint64{tm.q})
		if err != nil {
			fmt.Printf("  SKIP q=%d (%s): %v\n", tm.q, tm.label, err)
			continue
		}

		dist := ring.DiscreteGaussian{Sigma: 3.2, Bound: 19.2}
		sampler := ring.NewGaussianSampler(prng, rTest, dist, false)
		label := fmt.Sprintf("D_%s", tm.label)

		var trialResults []TestResult
		for t := 0; t < Trials; t++ {
			pol := sampler.ReadNew()
			directCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				directCoeffs[i] = coeffToSigned(pol.Coeffs[0][i], tm.q)
			}

			nttPol := rTest.NewPoly()
			copy(nttPol.Coeffs[0], pol.Coeffs[0])
			rTest.NTT(nttPol, nttPol)

			nttCoeffs := make([]float64, N)
			for i := 0; i < N; i++ {
				nttCoeffs[i] = coeffToSigned(nttPol.Coeffs[0][i], tm.q)
			}

			dk := kurtosis(directCoeffs)
			nk := kurtosis(nttCoeffs)
			tr := TestResult{Test: "D", Label: label, N: N, LogQ: tm.logQ, Modulus: tm.q, Trial: t + 1, DirectK: dk, NTTK: nk, AbsNTTK: math.Abs(nk)}
			trialResults = append(trialResults, tr)
			allResults = append(allResults, tr)
		}

		aggregate("D", label, N, fmt.Sprintf("log2(q)=%d", tm.logQ), trialResults)
		fmt.Printf("  log2(q)=%2d (q=%d): κ=%.4f±%.4f\n",
			tm.logQ, tm.q, mean(extract(trialResults)), stddev(extract(trialResults)))
	}
	fmt.Println()

	// ═══════════════════════════════════════════════════════════
	// TEST E: Plain DFT comparison (no ring structure)
	// ═══════════════════════════════════════════════════════════
	fmt.Println("▸ TEST E: Plain DFT vs cyclotomic NTT...")

	dist := ring.DiscreteGaussian{Sigma: 3.2, Bound: 19.2}
	sampler := ring.NewGaussianSampler(prng, r1024, dist, false)

	var dftKurtVals []float64
	var nttKurtVals []float64

	for t := 0; t < Trials; t++ {
		pol := sampler.ReadNew()

		// Get signed coefficients
		coeffs := make([]float64, N)
		for i := 0; i < N; i++ {
			coeffs[i] = coeffToSigned(pol.Coeffs[0][i], baseQ)
		}

		// Plain DFT (complex FFT magnitudes)
		dftResult := plainDFT(coeffs)
		dftKurt := kurtosis(dftResult)
		dftKurtVals = append(dftKurtVals, dftKurt)

		// Cyclotomic NTT
		nttPol := r1024.NewPoly()
		copy(nttPol.Coeffs[0], pol.Coeffs[0])
		r1024.NTT(nttPol, nttPol)
		nttCoeffs := make([]float64, N)
		for i := 0; i < N; i++ {
			nttCoeffs[i] = coeffToSigned(nttPol.Coeffs[0][i], baseQ)
		}
		nttKurt := kurtosis(nttCoeffs)
		nttKurtVals = append(nttKurtVals, nttKurt)

		tr1 := TestResult{Test: "E_DFT", Label: "E_plainDFT", N: N, Trial: t + 1, NTTK: dftKurt, AbsNTTK: math.Abs(dftKurt)}
		tr2 := TestResult{Test: "E_NTT", Label: "E_cyclotomicNTT", N: N, Trial: t + 1, NTTK: nttKurt, AbsNTTK: math.Abs(nttKurt)}
		allResults = append(allResults, tr1, tr2)
	}

	fmt.Printf("  Plain DFT (magnitude): κ=%.4f±%.4f\n", mean(dftKurtVals), stddev(dftKurtVals))
	fmt.Printf("  Cyclotomic NTT:        κ=%.4f±%.4f\n", mean(nttKurtVals), stddev(nttKurtVals))

	aggResults = append(aggResults, AggResult{
		Test: "E", Label: "E_plainDFT", N: N, Param: "DFT_magnitude",
		MeanK: mean(dftKurtVals), StdK: stddev(dftKurtVals), Trials: Trials,
	})
	aggResults = append(aggResults, AggResult{
		Test: "E", Label: "E_cyclotomicNTT", N: N, Param: "NTT_cyclotomic",
		MeanK: mean(nttKurtVals), StdK: stddev(nttKurtVals), Trials: Trials,
	})
	fmt.Println()

	// ═══════════════════════════════════════════════════════════
	// SUMMARY
	// ═══════════════════════════════════════════════════════════
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("ROUND 2 SUMMARY")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Printf("%-25s  %10s  %10s  %8s\n", "Test", "Mean κ", "Std", "Trials")
	fmt.Println("─────────────────────────────────────────────────────────")
	for _, a := range aggResults {
		fmt.Printf("%-25s  %10.4f  %10.4f  %8d\n", a.Label+"("+a.Param+")", a.MeanK, a.StdK, a.Trials)
	}
	fmt.Println()
	fmt.Printf("Reference: -6/5 = %.6f  |  -√6/2 = %.6f  |  -√1.25 = %.6f\n",
		-6.0/5.0, -math.Sqrt(6)/2, -math.Sqrt(1.25))
	fmt.Println()

	// ── Save ──
	output := map[string]interface{}{
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
		"constants": map[string]float64{
			"neg_6_over_5":  -6.0 / 5.0,
			"neg_sqrt6_2":   -math.Sqrt(6) / 2,
			"neg_sqrt_1_25": -math.Sqrt(1.25),
		},
		"aggregates": aggResults,
		"all_results": allResults,
	}
	jsonData, _ := json.MarshalIndent(output, "", "  ")
	os.WriteFile(OutputDir+"/round2_results.json", jsonData, 0644)

	// CSV
	csvFile, _ := os.Create(OutputDir + "/round2_results.csv")
	cw := csv.NewWriter(csvFile)
	cw.Write([]string{"test", "label", "N", "sigma", "log_q", "modulus", "trial", "direct_kurtosis", "ntt_kurtosis", "abs_ntt_kurtosis"})
	for _, r := range allResults {
		cw.Write([]string{
			r.Test, r.Label, fmt.Sprintf("%d", r.N),
			fmt.Sprintf("%.1f", r.Sigma), fmt.Sprintf("%d", r.LogQ),
			fmt.Sprintf("%d", r.Modulus), fmt.Sprintf("%d", r.Trial),
			fmt.Sprintf("%.6f", r.DirectK), fmt.Sprintf("%.6f", r.NTTK),
			fmt.Sprintf("%.6f", r.AbsNTTK),
		})
	}
	cw.Flush()
	csvFile.Close()

	fmt.Printf("Results: %s/round2_results.json\n", OutputDir)
	fmt.Printf("CSV:     %s/round2_results.csv\n", OutputDir)
	fmt.Printf("KS data: %s/ks_ntt_N*.csv\n", OutputDir)
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("Now run: python3 round2_analysis.py")
}

// plainDFT computes a naive DFT and returns magnitudes
func plainDFT(x []float64) []float64 {
	n := len(x)
	result := make([]float64, n)
	for k := 0; k < n; k++ {
		var sum complex128
		for j := 0; j < n; j++ {
			angle := -2 * math.Pi * float64(k) * float64(j) / float64(n)
			sum += complex(x[j], 0) * cmplx.Rect(1, angle)
		}
		result[k] = cmplx.Abs(sum) / float64(n)
	}
	return result
}

func extract(results []TestResult) []float64 {
	v := make([]float64, len(results))
	for i, r := range results {
		v[i] = r.NTTK
	}
	return v
}

func extractDirect(results []TestResult) []float64 {
	v := make([]float64, len(results))
	for i, r := range results {
		v[i] = r.DirectK
	}
	return v
}
