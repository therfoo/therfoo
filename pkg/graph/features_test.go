package graph

import (
	"math/rand"
	"strconv"
	"testing"
)

func TestFeaturesClassWeights(t *testing.T) {
	for k, v := range []struct {
		samples  int
		classes  int
		features int
		minority int
		weights  []float64
	}{
		{43400, 5, 21, 783, []float64{0.8159428463996992, 0.8102305610006534, 0.8234512854567878, 0.8093240093240093, 11.085568326947637}},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			features := Features{
				X: make([][]float64, v.samples),
				Y: make([][]float64, v.samples),
			}
			rand.Seed(80)
			n := v.classes - 1
			for i := 0; i < v.samples; i++ {
				features.X[i] = make([]float64, v.features)
				features.Y[i] = make([]float64, v.classes)
				if i < v.minority {
					features.Y[i][n] = 1
				} else {
					features.Y[i][rand.Intn(n)] = 1
				}
			}
			features.Shuffle()
			want := v.weights
			features.Balance()
			got := features.ClassWeights
			for i := range want {
				if want[i]-got[i] > 1e-9 {
					t.Fatalf("want[%d] %g, got[%d] %g", i, want, i, got)
				}
			}
		})
	}
}

func TestFeaturesShuffle(t *testing.T) {
	for k, v := range []struct {
		features Features
		want     Features
	}{
		{
			Features{
				X: [][]float64{{0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}},
				Y: [][]float64{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}},
			},
			Features{
				X: [][]float64{{0.5}, {0.9}, {0.1}, {0.7}, {0.8}, {0.4}, {0}, {0.2}, {0.6}, {0.3}},
				Y: [][]float64{{1, 0}, {1, 0}, {0, 1}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {0, 1}},
			},
		},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			v.features.Shuffle()
			got := v.features.X
			want := v.want.X
			for i := range want {
				for j := range want[i] {
					if want[i][j] != got[i][j] {
						t.Fatalf("want %v, got %v", want, got)
					}
				}
			}
			got = v.features.Y
			want = v.want.Y
			for i := range want {
				for j := range want[i] {
					if want[i][j] != got[i][j] {
						t.Fatalf("want %v, got %v", want, got)
					}
				}
			}
		})
	}
}

func TestFeaturesSplit(t *testing.T) {
	for k, v := range []struct {
		data       Features
		training   Features
		validating Features
	}{
		{
			Features{
				X: [][]float64{{0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}},
				Y: [][]float64{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}},
			},
			Features{
				X: [][]float64{{0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}},
				Y: [][]float64{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}},
			},
			Features{
				X: [][]float64{{0.8}, {0.9}},
				Y: [][]float64{{1, 0}, {1, 0}},
			},
		},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			training, validating := v.data.Split(0.2)
			want := v.training.X
			got := training.X
			for i := range want {
				for j := range want[i] {
					if want[i][j] != got[i][j] {
						t.Errorf("want %v, got %v", want, got)
					}
				}
			}

			want = v.validating.Y
			got = validating.Y
			for i := range want {
				for j := range want[i] {
					if want[i][j] != got[i][j] {
						t.Fatalf("want %v, got %v", want, got)
					}
				}
			}
		})
	}
}
