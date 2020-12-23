package graph

type ConstantLearningRate float64

func (r ConstantLearningRate) Rate() float64 {
	return float64(r)
}
