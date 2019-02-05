package sgd

import (
	"github.com/therfoo/therfoo/tensor"
)

type Operation func(*SGD)

type SGD struct {
	batchSize    int
	deltas       [][][]float64
	learningRate float64
	operations   chan Operation
}

func (o *SGD) Add(layer int, activation, cost *tensor.Vector) {
	o.operations <- func(d *SGD) {
		for n := range o.deltas[layer] {
			for w := range o.deltas[layer][n] {
				if w == 0 {
					o.deltas[layer][n][w] += (*cost)[n]
				} else {
					o.deltas[layer][n][w] += (*activation)[w-1] * (*cost)[n]
				}
			}
		}
	}
}

func (o *SGD) Init(neurons *[][]int) {
	layers := len(*neurons)
	o.deltas = make([][][]float64, layers, layers)
	for l := range o.deltas {
		o.deltas[l] = make([][]float64, (*neurons)[l][0], (*neurons)[l][0])
		for n := range o.deltas[l] {
			o.deltas[l][n] = make([]float64, (*neurons)[l][1]+1, (*neurons)[l][1]+1)
		}
	}
}

func (o *SGD) Optimizations() *[][][]float64 {
	done := make(chan struct{})

	o.operations <- func(d *SGD) {
		for l := range d.deltas {
			for n := range d.deltas[l] {
				for p := range d.deltas[l][n] {
					d.deltas[l][n][p] = d.learningRate * (d.deltas[l][n][p] / float64(d.batchSize))
				}
			}
		}
		done <- struct{}{}
	}

	<-done

	return &o.deltas
}

func (o *SGD) work() {
	for op := range o.operations {
		op(o)
	}
}

func New(options ...Option) *SGD {
	o := SGD{operations: make(chan Operation)}

	for i := range options {
		options[i](&o)
	}

	go o.work()

	return &o
}
