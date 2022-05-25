package graph

import (
	"math/rand"
)

type Features struct {
	ClassWeights        []float64
	DisableClassWeights bool
	DisableShuffle      bool
	X                   [][]float64
	Y                   [][]float64
}

func (f *Features) Balance() {
	if len(f.Y) == 0 || len(f.Y[0]) == 0 {
		return
	}
	n := len(f.Y)
	m := len(f.Y[0])
	f.ClassWeights = make([]float64, m)
	for i := range f.Y {
		f.ClassWeights[argmax(f.Y[i])]++
	}
	for i := range f.ClassWeights {
		f.ClassWeights[i] = float64(n) / (float64(m) * f.ClassWeights[i])
	}
}

func (f *Features) Prepare() {
	if !f.DisableShuffle {
		f.Shuffle()
	}
	if !f.DisableClassWeights {
		f.Balance()
	}
}

func (f *Features) Shuffle() {
	r := rand.New(rand.NewSource(DefaultSeed))
	r.Shuffle(len(f.X), func(i, j int) {
		f.X[i], f.X[j] = f.X[j], f.X[i]
		f.Y[i], f.Y[j] = f.Y[j], f.Y[i]
	})
}

func (f *Features) Split(split float64) (training, validation Features) {
	if 0 > split || split > 1 {
		panic("validationSplit must be a float between 0 and 1")
	}
	n := int64((1 - split) * float64(len(f.X)))
	return Features{X: f.X[:n], Y: f.Y[:n]}, Features{X: f.X[n:], Y: f.Y[n:]}
}
