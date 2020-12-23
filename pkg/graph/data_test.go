package graph

import (
	"math/rand"
	"strconv"
	"testing"
)

func TestDataClassWeights(t *testing.T) {
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
			data := Data{
				X: make([][]float64, v.samples),
				Y: make([][]float64, v.samples),
			}
			rand.Seed(80)
			n := v.classes - 1
			for i := 0; i < v.samples; i++ {
				data.X[i] = make([]float64, v.features)
				data.Y[i] = make([]float64, v.classes)
				if i < v.minority {
					data.Y[i][n] = 1
				} else {
					data.Y[i][rand.Intn(n)] = 1
				}
			}
			data.Shuffle()
			want := v.weights
			data.Balance()
			got := data.ClassWeights
			for i := range want {
				if want[i]-got[i] > 1e-9 {
					t.Fatalf("want[%d] %g, got[%d] %g", i, want, i, got)
				}
			}
		})
	}
}

func TestDataShuffle(t *testing.T) {
	for k, v := range []struct {
		data Data
		want Data
	}{
		{
			Data{
				X: [][]float64{{0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}},
				Y: [][]float64{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}},
			},
			Data{
				X: [][]float64{{0.5}, {0.9}, {0.1}, {0.7}, {0.8}, {0.4}, {0}, {0.2}, {0.6}, {0.3}},
				Y: [][]float64{{1, 0}, {1, 0}, {0, 1}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {0, 1}},
			},
		},
	} {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			v.data.Shuffle()
			got := v.data.X
			want := v.want.X
			for i := range want {
				for j := range want[i] {
					if want[i][j] != got[i][j] {
						t.Fatalf("want %v, got %v", want, got)
					}
				}
			}
			got = v.data.Y
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

func TestDataSplit(t *testing.T) {
	for k, v := range []struct {
		data       Data
		training   Data
		validating Data
	}{
		{
			Data{
				X: [][]float64{{0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}},
				Y: [][]float64{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}},
			},
			Data{
				X: [][]float64{{0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}},
				Y: [][]float64{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}},
			},
			Data{
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
