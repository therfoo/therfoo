package layers

import (
	"math"
)

type Conv struct {
	Filters, FilterHeight, FilterWidth, Stride int
	inputShape, outputShape                    Shape
	outputs, weights                           [][]float64
	horizontal, vertical                       int
}

func (l *Conv) convolve(features []float64) {
	width := int(l.inputShape[1])
	for h := 0; h < l.Filters; h++ {
		for i := 0; i < l.vertical; i += l.Stride {
			for j := 0; j < l.horizontal; j += l.Stride {
				for k := 0; k < l.FilterHeight; k++ {
					for m := 0; m < l.FilterWidth; m++ {
						z := j + i*l.horizontal
						a := m + k*width + j + i*width
						b := m + k*l.FilterWidth
						l.outputs[h][z] = math.FMA(features[a], l.weights[h][b], l.outputs[h][z])
					}
				}
			}
		}
	}
}

func (l *Conv) Estimate(features []float64) []float64 {
	l.convolve(features)
	return features
}

func (l *Conv) SetShape(shape []uint64) {
	l.inputShape = shape
	l.outputShape = append(shape, uint64(l.Filters))
	l.init()
}

func (l *Conv) init() {
	s := l.inputShape
	l.outputs, l.weights = make([][]float64, l.Filters), make([][]float64, l.Filters)
	l.horizontal, l.vertical = s.Width()-l.FilterWidth+1, s.Height()-l.FilterHeight+1
	size := l.horizontal * l.vertical
	for k := range l.outputs {
		l.outputs[k] = make([]float64, size)
	}
	size = l.FilterHeight * l.FilterWidth
	for k := range l.weights {
		l.weights[k] = Random(size)
	}
}

func (l *Conv) Shape() []uint64 {
	return l.outputShape.Shape()
}

func (l *Conv) Minimize(gradients []float64) []float64 {
	return gradients
}
