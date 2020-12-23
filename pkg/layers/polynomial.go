package layers

import (
	"fmt"
	"math"
	"strings"
)

type Polynomial struct {
	layer
	learner
	Degree int
	input  []float64
	terms  [][]float64
}

func (l *Polynomial) Estimate(input []float64) []float64 {
	copy(l.input, input)
	for j := range l.terms {
		var p float64
		for k := range l.terms[j] {
			l.terms[j][k] = math.Pow(input[j], float64(k))
			p = math.FMA(l.weights[j][k], l.terms[j][k], p)
		}
		l.output[j] = input[j] * p
	}
	return l.output
}

func (l *Polynomial) Minimize(gradients []float64) []float64 {
	for j := range l.weights {
		var g float64
		for k := range l.weights[j] {
			d := float64(k + 1)
			g = math.FMA(d*l.weights[j][k], l.terms[j][k], g)
			l.localGradients[j][k] = gradients[j] * l.input[j] * l.terms[j][k]
		}
		gradients[j] = gradients[j] * g
	}
	return gradients
}

func (l *Polynomial) SetShape(shape []uint64) {
	l.layer.SetShape(shape)
	n := l.outputShape.Size()
	l.input = make([]float64, n)
	l.localGradients = make([][]float64, n)
	l.terms = make([][]float64, n)
	l.weights = make([][]float64, n)
	for j := 0; j < n; j++ {
		l.localGradients[j] = make([]float64, l.Degree)
		l.terms[j] = make([]float64, l.Degree)
		l.weights[j] = Random(l.Degree)
	}
}

func (l *Polynomial) String() string {
	return l.toYAML()
}

func (l *Polynomial) toYAML() string {
	var s []string
	s = append(s, "polynomial:")
	s = append(s, fmt.Sprintf("%sgradients:", indent))
	for _, v := range l.localGradients {
		s = append(s, fmt.Sprintf("%s%s- %g", indent, indent, v))
	}
	s = append(s, fmt.Sprintf("%soutputs:", indent))
	for _, v := range l.output {
		s = append(s, fmt.Sprintf("%s%s- %g", indent, indent, v))
	}
	s = append(s, fmt.Sprintf("%spolynomials:", indent))
	for j := range l.terms {
		s = append(s, fmt.Sprintf("%s%s-", indent, indent))
		s = append(s, fmt.Sprintf("%s%sinput: %g", indent, indent, l.input[j]))
		s = append(s, fmt.Sprintf("%s%scoefficients:", indent, indent))
		for _, v := range l.weights[j] {
			s = append(s, fmt.Sprintf("%s%s%s- %g", indent, indent, indent, v))
		}
		s = append(s, fmt.Sprintf("%s%sterms:", indent, indent))
		for _, v := range l.terms[j] {
			s = append(s, fmt.Sprintf("%s%s%s- %g", indent, indent, indent, v))
		}
	}
	return strings.Join(s, eol)
}
