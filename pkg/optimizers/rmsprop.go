package optimizers

import "math"

type RMSprop struct {
	gradients [][]float64
	momentum  float64
}

func (m RMSprop) Apply(weights [][]float64) RMSprop {
	gradients := make([][]float64, len(weights))
	for i := range gradients {
		gradients[i] = make([]float64, len(weights[i]))
	}
	return RMSprop{
		gradients: gradients,
		momentum:  m.momentum,
	}
}

func (m RMSprop) Optimize(gradients [][]float64) [][]float64 {
	for i := range gradients {
		for j := range gradients[i] {
			m.gradients[i][j] = m.momentum*m.gradients[i][j] + (1-m.momentum)*math.Pow(gradients[i][j], 2)
		}
	}
	return m.gradients
}
