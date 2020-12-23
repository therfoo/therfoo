package regularizers

import "math"

type Ridge struct {
	Lambda  float64
	weights [][]float64
}

func (r Ridge) Apply(weights [][]float64) Ridge {
	return Ridge{Lambda: r.Lambda, weights: weights}
}

func (r Ridge) Regularize(gradients [][]float64) {
	if r.Lambda == 0 {
		return
	}
	for i := range r.weights {
		for j := range r.weights[i] {
			gradients[i][j] = math.FMA(r.Lambda, math.Pow(r.weights[i][j], 2), gradients[i][j])
		}
	}
}

var DefaultRidge = Ridge{Lambda: 0.01}
