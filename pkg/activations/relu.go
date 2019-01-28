package activations

import (
	"github.com/therfoo/therfoo/pkg/tensor"
)

func ReLU(z *tensor.Vector) *tensor.Vector {
	n := z.Len()
	a := make(tensor.Vector, n, n)

	z.Each(func(index int, value float64) {
		if value >= 0. {
			a[index] = value
		} else {
			a[index] = 0.
		}
	})

	return &a
}

func ReLUPrime(a *tensor.Vector) *tensor.Vector {
	n := a.Len()
	d := make(tensor.Vector, n, n)

	a.Each(func(index int, value float64) {
		if value > 0. {
			d[index] = 1.
		} else {
			d[index] = 0.
		}
	})

	return &d
}
