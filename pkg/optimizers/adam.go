package optimizers

import "math"

type Adam struct {
	epsilon   float64
	gradients [][]float64
	momentum  Momentum
	rmsprop   RMSprop
}

func (m Adam) Apply(weights [][]float64) Adam {
	gradients := make([][]float64, len(weights))
	for i := range gradients {
		gradients[i] = make([]float64, len(weights[i]))
	}
	return Adam{
		gradients: gradients,
	}
}

func (m Adam) Optimize(gradients [][]float64) [][]float64 {
	v := m.momentum.Optimize(gradients)
	s := m.rmsprop.Optimize(gradients)
	for i := range m.gradients {
		for j := range m.gradients[i] {
			m.gradients[i][j] = (v[i][j] / (math.Sqrt(s[i][j] + m.epsilon)))
		}
	}
	return m.gradients
}
