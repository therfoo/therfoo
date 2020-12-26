package layers

import (
	"fmt"
	"math"
	"strings"
)

type UnbiasedDense struct {
	layer
	learner
	gradients []float64
	input     []float64
	Neurons   uint64
}

func (d *UnbiasedDense) Estimate(input []float64) []float64 {
	d.input = input
	for j := range d.weights {
		var z float64
		for k := range d.weights[j] {
			z = math.FMA(d.weights[j][k], input[k], z)
		}
		d.output[j] = z
	}
	return d.output
}

func (d *UnbiasedDense) Minimize(gradients []float64) []float64 {
	for k := range d.gradients {
		d.gradients[k] = 0
	}
	for j := range d.weights {
		for k := range d.weights[j] {
			d.localGradients[j][k] = gradients[j] * d.input[k]
			d.gradients[k] += gradients[j] * d.weights[j][k]
		}
	}
	return d.gradients
}

func (d *UnbiasedDense) SetShape(shape []uint64) {
	d.inputShape = shape
	d.outputShape = Shape{d.Neurons}
	w := d.inputShape.Size()
	n := d.Neurons
	d.input = make([]float64, w)
	d.output = make([]float64, n)
	d.gradients = make([]float64, w)
	d.localGradients = make([][]float64, n)
	d.weights = make([][]float64, n)
	for j := range d.weights {
		d.localGradients[j] = make([]float64, w)
		d.weights[j] = Random(w)
	}
}

func (d *UnbiasedDense) String() string {
	var s []string
	s = append(s, "dense:")
	s = append(s, fmt.Sprintf("%sinputs:", indent))
	for _, v := range d.input {
		s = append(s, fmt.Sprintf("%s%s- %g", indent, indent, v))
	}
	s = append(s, fmt.Sprintf("%soutputs:", indent))
	for _, v := range d.output {
		s = append(s, fmt.Sprintf("%s%s- %g", indent, indent, v))
	}
	s = append(s, fmt.Sprintf("%sweights:", indent))
	for j := range d.weights {
		s = append(s, fmt.Sprintf("%s%s-", indent, indent))
		for k := range d.weights[j] {
			s = append(s, fmt.Sprintf("%s%s%s- %g", indent, indent, indent, d.weights[j][k]))
		}
	}
	return strings.Join(s, eol)
}
