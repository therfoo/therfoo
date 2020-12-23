package layers

import (
	"fmt"
	"math"
	"strings"
)

type Sigmoid struct {
	layer
}

func (s *Sigmoid) Activate(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (s *Sigmoid) Derive(a float64) float64 {
	return a * (1.0 - a)
}

func (s *Sigmoid) Estimate(x []float64) []float64 {
	for i := range x {
		s.output[i] = s.Activate(x[i])
	}
	return s.output
}

func (s *Sigmoid) Minimize(loss []float64) []float64 {
	for i := range loss {
		loss[i] = loss[i] * s.Derive(s.output[i])
	}
	return loss
}

func (s *Sigmoid) String() string {
	lines := []string{"sigmoid:"}
	lines = append(lines, fmt.Sprintf("%soutputs:", indent))
	for _, v := range s.output {
		lines = append(lines, fmt.Sprintf("%s%s- %g", indent, indent, v))
	}
	return strings.Join(lines, eol)
}
