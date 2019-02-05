package optimizers

import (
	"github.com/therfoo/therfoo/tensor"
)

type Optimizer interface {
	Add(layer int, cost, activation *tensor.Vector)
	Init(deltas *[][]int)
	Optimizations() *[][][]float64
}
