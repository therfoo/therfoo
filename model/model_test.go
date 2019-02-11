package model

import (
	"github.com/therfoo/therfoo/data/mnist"
	"github.com/therfoo/therfoo/layers/dense"
	"github.com/therfoo/therfoo/optimizers/sgd"
	"testing"
)

func TestFit(t *testing.T) {
	lr := 0.005
	m := New(
		WithEpochs(100),
		WithInputShape([]int{28, 28}),
		WithOptimizer(
			sgd.New(
				sgd.WithBatchSize(100),
				sgd.WithLearningRate(lr),
			),
		),
		WithCrossEntropyLoss(),
		WithTestingGenerator(mnist.New(mnist.WithBatchSize(100), mnist.WithTesting())),
		WithTrainingGenerator(mnist.New(mnist.WithBatchSize(100), mnist.WithTesting())),
		WithValidatingGenerator(mnist.New(mnist.WithBatchSize(100), mnist.WithTesting())),
	)

	m.Add(28, dense.New(dense.WithReLU()))
	m.Add(10, dense.New(dense.WithSigmoid()))

	m.Compile()

	trainingMetrics := make(chan *TrainingMetrics, 10)

	go func() {
		for metric := range trainingMetrics {
			t.Log(metric.Metrics.Cost)
		}
	}()

	m.Fit(trainingMetrics)

}
