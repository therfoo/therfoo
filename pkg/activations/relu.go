package activations

import (
	"github.com/therfoo/therfoo/pkg/tensor"
)

func ReLU(z *tensor.Vector) *tensor.Vector {
	a := tensor.Vector{}

	z.Each(func(index int, value float64) {
		if value >= 0. {
			a.Append(value)
		} else {
			a.Append(0.)
		}
	})

	return &a
}

func ReLUPrime(a *tensor.Vector) *tensor.Vector {
	d := tensor.Vector{}

	a.Each(func(index int, value float64) {
		if value > 0 {
			d.Append(1)
		} else {
			d.Append(0)
		}
	})

	return &d
}
