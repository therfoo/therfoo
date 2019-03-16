package sgd

import (
	"github.com/therfoo/therfoo/tensor"
	"sync"
)

type Operation func(*SGD)

type SGD struct {
	batchSize    int
	deltas       [][][]float64
	learningRate float64
	mutex        sync.Mutex
}

func (o *SGD) Add(layer int, activation, cost *tensor.Vector) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
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
	o.mutex.Lock()
	defer o.mutex.Unlock()
	for l := range o.deltas {
		for n := range o.deltas[l] {
			for p := range o.deltas[l][n] {
				o.deltas[l][n][p] = o.learningRate * (o.deltas[l][n][p] / float64(o.batchSize))
			}
		}
	}
	return &o.deltas
}

func New(options ...Option) *SGD {
	o := SGD{}

	for i := range options {
		options[i](&o)
	}

	return &o
}
