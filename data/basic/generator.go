package basic

import (
	"github.com/therfoo/therfoo/tensor"
)

type Generator struct{}

func (g *Generator) Get(index int) (x, y *[]tensor.Vector) {
	data := data[index]
	return &[]tensor.Vector{data.features}, &[]tensor.Vector{data.labels}
}

func (g *Generator) Len() int {
	return len(data)
}

func New() *Generator {
	return &Generator{}
}
