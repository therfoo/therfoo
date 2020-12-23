package graph

import (
	"fmt"
	"strings"
)

type Metrics struct {
	Actual   []float64
	Estimate []float64
	Epoch    int
	Loss     []float64
	Sample   int
}

func (m Metrics) String() string {
	var str = func(f []float64) string {
		s := make([]string, len(f))
		for k := range f {
			s[k] = fmt.Sprintf("%.4f", f[k])
		}
		return fmt.Sprintf("[%s]", strings.Join(s, ", "))
	}
	var loss = func(a, y []float64) float64 {
		var sum float64
		for k := range a {
			sum += a[k] - y[k]
		}
		return sum
	}
	return fmt.Sprintf("epoch: %d, loss: %.4f, estimate: %s, actual: %s", m.Epoch, loss(m.Estimate, m.Actual), str(m.Estimate), str(m.Actual))
}
