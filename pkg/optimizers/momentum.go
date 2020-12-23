package optimizers

type Momentum struct {
	gradients [][]float64
	momentum  float64
}

func (m Momentum) Apply(weights [][]float64) Momentum {
	gradients := make([][]float64, len(weights))
	for i := range gradients {
		gradients[i] = make([]float64, len(weights[i]))
	}
	return Momentum{
		gradients: gradients,
		momentum:  m.momentum,
	}
}

func (m Momentum) Optimize(gradients [][]float64) [][]float64 {
	for i := range gradients {
		for j := range gradients[i] {
			m.gradients[i][j] = m.momentum*m.gradients[i][j] + (1-m.momentum)*gradients[i][j]
		}
	}
	return m.gradients
}
