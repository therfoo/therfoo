package layers

type learner struct {
	weights        [][]float64
	localGradients [][]float64
}

func (l learner) Gradients() [][]float64 {
	return l.localGradients
}

func (l learner) Weights() [][]float64 {
	return l.weights
}
