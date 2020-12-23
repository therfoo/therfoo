package layers

type BiasedDense struct {
	bias    Bias
	dense   Dense
	Neurons uint64
}

func (l *BiasedDense) Estimate(input []float64) []float64 {
	input = l.dense.Estimate(input)
	return l.bias.Estimate(input)
}

func (l *BiasedDense) Gradients() [][]float64 {
	return append(l.bias.localGradients, l.dense.localGradients...)
}

func (l *BiasedDense) Minimize(gradients []float64) []float64 {
	gradients = l.bias.Minimize(gradients)
	return l.dense.Minimize(gradients)
}

func (l *BiasedDense) SetShape(shape []uint64) {
	l.dense.Neurons = l.Neurons
	l.dense.SetShape(shape)
	l.bias.SetShape(l.dense.Shape())
}

func (l *BiasedDense) Shape() []uint64 {
	return l.bias.Shape()
}

func (l *BiasedDense) String() string {
	return l.dense.String() + eol + l.bias.String()
}

func (l *BiasedDense) Weights() [][]float64 {
	return append(l.bias.weights, l.dense.weights...)
}
