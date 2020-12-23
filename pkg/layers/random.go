package layers

import (
	"math/rand"
)

func Random(size int) []float64 {
	w := make([]float64, size)
	r := rand.New(rand.NewSource(9868))
	for i := range w {
		w[i] = r.NormFloat64()
	}
	return w
}
