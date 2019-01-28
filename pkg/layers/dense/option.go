package dense

import (
	"github.com/therfoo/therfoo/pkg/activations"
)

type Option func(*Dense)

func WithReLU() Option {
	return func(d *Dense) {
		d.activate = activations.ReLU
		d.derive = activations.ReLUPrime
	}
}

func WithLeakyReLU() Option {
	return func(d *Dense) {
		d.activate = activations.LeakyReLU
		d.derive = activations.LeakyReLUPrime
	}
}

func WithSigmoid() Option {
	return func(d *Dense) {
		d.activate = activations.Sigmoid
		d.derive = activations.SigmoidPrime
	}
}

func WithSoftmax() Option {
	return func(d *Dense) {
		d.activate = activations.Softmax
		d.derive = activations.SoftmaxPrime
	}
}
