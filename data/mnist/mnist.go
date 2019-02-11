package mnist

import (
	"github.com/therfoo/therfoo/tensor"
)

type MNIST struct {
	indices   [][2]int
	batchSize int
	images    []tensor.Vector
	labels    []tensor.Vector
}

func (m *MNIST) index() {
	n := len(m.images)
	r := n % m.batchSize
	b := n / m.batchSize
	c := b
	if r > 0 {
		c++
	}
	m.indices = make([][2]int, c, c)
	for i := 0; i < b; i++ {
		m.indices[i] = [2]int{i * m.batchSize, (i + 1) * m.batchSize}
	}
	if r > 0 {
		m.indices[b] = [2]int{b * m.batchSize, n}
	}
}

func (m *MNIST) Len() int {
	return len(m.indices)
}

func (m *MNIST) Get(index int) (x, y *[]tensor.Vector) {
	a, z := m.indices[index][0], m.indices[index][1]
	images, labels := m.images[a:z], m.labels[a:z]
	return &images, &labels
}

func New(options ...Option) *MNIST {
	m := MNIST{}
	for i := range options {
		options[i](&m)
	}
	m.index()
	return &m
}
