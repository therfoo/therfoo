package graph

type Layer interface {
	Estimate([]float64) []float64
	Minimize([]float64) []float64
	SetShape([]uint64)
	Shape() []uint64
}
