package regularizers

import "math"

type Lasso struct {
	Lambda  float64
	weights [][]float64
}

func (r Lasso) Regularize(gradients [][]float64) {
	if r.Lambda == 0 {
		return
	}
	for i := range r.weights {
		for j := range r.weights[i] {
			gradients[i][j] = math.FMA(r.Lambda, math.Abs(r.weights[i][j]), gradients[i][j])
		}
	}
}

var DefaultLasso = Lasso{Lambda: 0.01}
