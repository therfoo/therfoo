package layers

type Dense struct {
	bias    Bias
	dense   UnbiasedDense
	Neurons uint64
}

func (l *Dense) Estimate(input []float64) []float64 {
	input = l.dense.Estimate(input)
	return l.bias.Estimate(input)
}

func (l *Dense) Gradients() [][]float64 {
	return append(l.bias.localGradients, l.dense.localGradients...)
}

func (l *Dense) Minimize(gradients []float64) []float64 {
	gradients = l.bias.Minimize(gradients)
	return l.dense.Minimize(gradients)
}

func (l *Dense) SetShape(shape []uint64) {
	l.dense.Neurons = l.Neurons
	l.dense.SetShape(shape)
	l.bias.SetShape(l.dense.Shape())
}

func (l *Dense) Shape() []uint64 {
	return l.bias.Shape()
}

func (l *Dense) String() string {
	return l.dense.String() + eol + l.bias.String()
}

func (l *Dense) Weights() [][]float64 {
	return append(l.bias.weights, l.dense.weights...)
}
