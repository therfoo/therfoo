package graph

type Regularizer interface {
	Regularize([][]float64)
}
