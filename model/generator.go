package model

import (
	"github.com/therfoo/therfoo/tensor"
)

type Generator interface {
	Get(index int) (x, y *[]tensor.Vector)
	Len() int
}
