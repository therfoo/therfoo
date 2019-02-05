package dense

import (
	"github.com/therfoo/therfoo/activations"
)

type Option func(*Dense)

func WithReLU() Option {
	return func(d *Dense) {
		d.activate = activations.ReLU
		d.derive = activations.ReLUPrime
	}
}

func WithSigmoid() Option {
	return func(d *Dense) {
		d.activate = activations.Sigmoid
		d.derive = activations.SigmoidPrime
	}
}
