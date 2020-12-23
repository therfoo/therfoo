package graph

type Minimizeable interface {
	Gradients() [][]float64
	Weights() [][]float64
}
