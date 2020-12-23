package layers

type Input Shape

func (i Input) Estimate(input []float64) []float64 {
	return input
}

func (i Input) Minimize(gradients []float64) []float64 {
	return gradients
}

func (i Input) SetShape(input []uint64) {}

func (i Input) Shape() []uint64 {
	return Shape(i).Shape()
}

func (i Input) String() string {
	return "input:"
}
