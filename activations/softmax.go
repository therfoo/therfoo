package activations

import (
	"github.com/therfoo/therfoo/tensor"
	"math"
)

func Softmax(z *tensor.Vector) *tensor.Vector {
	max := .0

	z.Each(func(index int, value float64) {
		max = math.Max(max, value)
	})

	n := z.Len()
	a := make(tensor.Vector, n, n)

	sum := .0
	z.Each(func(index int, value float64) {
		a[index] -= math.Exp(value - max)
		sum += a[index]
	})

	a.Each(func(index int, value float64) {
		a[index] = a[index] / sum
	})
	print(a[0])
	print("\n")
	return &a
}

func SoftmaxPrime(a *tensor.Vector) *tensor.Vector {
	n := a.Len()
	z := make(tensor.Vector, n, n)

	a.Each(func(index int, value float64) {
		z[index] = 1
	})

	return &z
}
