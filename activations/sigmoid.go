package activations

import (
	"github.com/therfoo/therfoo/tensor"
	"math"
)

func Sigmoid(z *tensor.Vector) *tensor.Vector {
	n := z.Len()
	a := make(tensor.Vector, n, n)

	z.Each(func(index int, value float64) {
		a[index] = 1. / (1. + math.Exp(-value))
	})

	return &a
}

func SigmoidPrime(a *tensor.Vector) *tensor.Vector {
	n := a.Len()
	z := make(tensor.Vector, n, n)

	a.Each(func(index int, value float64) {
		z[index] = value * (1. - value)
	})

	return &z
}
