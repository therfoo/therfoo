package model

import (
	"github.com/therfoo/therfoo/tensor"
)

type Layer interface {
	Activate(input *tensor.Vector) (output *tensor.Vector)
	Bytes() ([]byte, error)
	Load(b []byte) error
	Derive(activation, cost *tensor.Vector)
	Adjust(delta *[][]float64)
	Init(neuronsCount, weightsCount int)
	NextCost(cost *tensor.Vector) (nextCost *tensor.Vector)
}
