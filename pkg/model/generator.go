package model

import (
	"github.com/therfoo/therfoo/pkg/tensor"
)

type Generator interface {
	Next(index int) (x, y *[]tensor.Vector)
	Len() int
}
