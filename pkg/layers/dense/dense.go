package dense

import (
	"github.com/therfoo/therfoo/pkg/tensor"
	"math/rand"
)

type Dense struct {
	activate     func(*tensor.Vector) *tensor.Vector
	derive       func(*tensor.Vector) *tensor.Vector
	neuronsCount int
	weightsCount int
	weights      [][]float64
}

func (d *Dense) Activate(x *tensor.Vector) tensor.Vector {
	z := tensor.Vector{}
	for neuron := range d.weights {
		sum := 0.
		for weight := range d.weights[neuron] {
			if weight == 0 {
				sum += d.weights[neuron][weight]
			} else {
				sum += d.weights[neuron][weight] * (*x)[weight-1]
			}
		}
		z.Append(sum)
	}

	return *d.activate(&z)
}

func (d *Dense) Derive(activation, cost *tensor.Vector) {
	d.derive(activation).Each(func(index int, value float64) {
		(*cost)[index] = (*cost)[index] * value
	})
}

func (d *Dense) Adjust(delta *[][]float64) {
	for n := range *delta {
		for p := range (*delta)[n] {
			d.weights[n][p] -= (*delta)[n][p]
		}
	}
}

func (d *Dense) Init(neuronsCount, weightsCount int) {
	d.neuronsCount = neuronsCount
	d.weightsCount = weightsCount
	d.weights = make([][]float64, d.neuronsCount, d.neuronsCount)
	totalWeights := d.weightsCount + 1
	for n := range d.weights {
		d.weights[n] = make([]float64, totalWeights, totalWeights)
		for w := range d.weights[n] {
			d.weights[n][w] = rand.Float64()
		}
	}
}

func (d *Dense) NextCost(cost *tensor.Vector) *tensor.Vector {
	next := make(tensor.Vector, d.weightsCount, d.weightsCount)
	for n := range d.weights {
		for w := range d.weights[n] {
			if w > 0 {
				next[w-1] += d.weights[n][w-1] * (*cost)[n]
			}
		}
	}
	return &next
}

func (d *Dense) Size() int {
	return d.neuronsCount
}

func New(options ...Option) *Dense {
	d := Dense{}
	for i := range options {
		options[i](&d)
	}
	return &d
}
