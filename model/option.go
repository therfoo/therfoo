package model

import (
	"github.com/therfoo/therfoo/losses"
	"github.com/therfoo/therfoo/optimizers"
)

type Option func(*Model)

func WithTestingGenerator(g Generator) Option {
	return func(m *Model) {
		m.generator.testing = g
	}
}

func WithTrainingGenerator(g Generator) Option {
	return func(m *Model) {
		m.generator.training = g
	}
}

func WithValidatingGenerator(g Generator) Option {
	return func(m *Model) {
		m.generator.validating = g
	}
}

func WithInputShape(shape []int) Option {
	return func(m *Model) {
		m.inputShape = shape
	}
}

func WithEpochs(epochs int) Option {
	return func(m *Model) {
		m.epochs = epochs
	}
}

func WithOptimizer(o optimizers.Optimizer) Option {
	return func(m *Model) {
		m.optimizer = o
	}
}

func WithCrossEntropyLoss() Option {
	return func(m *Model) {
		m.lossFunction = losses.CrossEntropy
		m.lossPrime = losses.CrossEntropyPrime
		m.skipOutputDerivative = true
	}
}
