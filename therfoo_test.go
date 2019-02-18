package therfoo

import (
	"github.com/therfoo/datasets/basic"

	"github.com/therfoo/therfoo/layers/dense"
	"github.com/therfoo/therfoo/model"
	"github.com/therfoo/therfoo/optimizers/sgd"
	"github.com/therfoo/therfoo/tensor"
	"testing"
)

func TestModelFit(t *testing.T) {
	m := model.New(
		model.WithBinaryAccuracy(),
		model.WithCrossEntropyLoss(),
		model.WithEpochs(25),
		model.WithInputShape(tensor.Shape{2}),
		model.WithOptimizer(
			sgd.New(sgd.WithBatchSize(1), sgd.WithLearningRate(0.05)),
		),
		model.WithTrainingGenerator(basic.New()),
		model.WithValidatingGenerator(basic.New()),
		model.WithVerbosity(true),
	)

	m.Add(4, dense.New(dense.WithReLU()))
	m.Add(1, dense.New(dense.WithSigmoid()))

	m.Compile()

	m.Fit()

}
