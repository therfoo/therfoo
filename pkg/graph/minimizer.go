package graph

type Minimizer struct {
	Layer
	cursor       int
	batch        [][][]float64
	gradients    [][]float64
	learningRate LearningRate
	optimizer    Optimizer
	regularizer  Regularizer
	weights      [][]float64
}

func (m *Minimizer) Minimize(gradients []float64) []float64 {
	gradients = m.Layer.Minimize(gradients)
	for i := range m.gradients {
		copy(m.batch[m.cursor][i], m.gradients[i])
	}
	m.cursor++
	if m.cursor < len(m.batch) {
		return gradients
	}
	m.cursor = 0
	m.average()
	m.regularize()
	m.optimize()
	lr := m.learningRate.Rate()
	for i := range m.batch[0] {
		for j := range m.batch[0][i] {
			m.weights[i][j] -= lr * m.batch[0][i][j]
		}
	}
	return gradients
}

func (m *Minimizer) average() {
	batchSize := len(m.batch)
	n := float64(batchSize)
	for i := range m.batch[0] {
		for j := range m.batch[0][i] {
			m.batch[0][i][j] = m.batch[0][i][j] / n
		}
	}
	for i := 1; i < int(batchSize); i++ {
		for j := range m.batch[0] {
			for k := range m.batch[0][j] {
				m.batch[0][j][k] += m.batch[i][j][k] / n
			}
		}
	}
}

func (m *Minimizer) optimize() {
	if m.optimizer == nil {
		return
	}
	g := m.optimizer.Optimize(m.batch[0])
	for i := range g {
		copy(m.batch[0][i], g[i])
	}
}

func (m *Minimizer) regularize() {
	if m.regularizer != nil {
		m.regularizer.Regularize(m.batch[0])
	}
}
