package layers

type layer struct {
	inputShape  Shape
	outputShape Shape
	output      []float64
}

func (l *layer) Minimize(loss []float64) []float64 {
	return loss
}

func (l *layer) Shape() []uint64 {
	return l.outputShape.Shape()
}

func (*layer) SetShape(shape []uint64) {
}

func (l *layer) SetInputShape(shape []uint64) {
	l.inputShape = shape
}

func (l *layer) SetOutputShape(shape []uint64) {
	l.outputShape = shape
	l.output = make([]float64, l.outputShape.Size())
}
