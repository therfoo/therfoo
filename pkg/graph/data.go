package graph

import (
	"math/rand"
)

type Data struct {
	ClassWeights        []float64
	DisableClassWeights bool
	DisableShuffle      bool
	X                   [][]float64
	Y                   [][]float64
}

func (d *Data) Balance() {
	if len(d.Y) == 0 || len(d.Y[0]) == 0 {
		return
	}
	n := len(d.Y)
	m := len(d.Y[0])
	d.ClassWeights = make([]float64, m)
	for i := range d.Y {
		d.ClassWeights[argmax(d.Y[i])]++
	}
	for i := range d.ClassWeights {
		d.ClassWeights[i] = float64(n) / (float64(m) * d.ClassWeights[i])
	}
}

func (c *Data) Prepare() {
	if !c.DisableShuffle {
		c.Shuffle()
	}
	if !c.DisableClassWeights {
		c.Balance()
	}
}

func (d *Data) Shuffle() {
	r := rand.New(rand.NewSource(DefaultSeed))
	r.Shuffle(len(d.X), func(i, j int) {
		d.X[i], d.X[j] = d.X[j], d.X[i]
		d.Y[i], d.Y[j] = d.Y[j], d.Y[i]
	})
}

func (d *Data) Split(split float64) (training, validation Data) {
	if 0 > split || split > 1 {
		panic("validationSplit must be a float between 0 and 1")
	}
	n := int64((1 - split) * float64(len(d.X)))
	return Data{X: d.X[:n], Y: d.Y[:n]}, Data{X: d.X[n:], Y: d.Y[n:]}
}
