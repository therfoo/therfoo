[![GoDoc](https://godoc.org/github.com/therfoo/therfoo?status.svg)](https://godoc.org/github.com/therfoo/therfoo)
[![Go Report Card](https://goreportcard.com/badge/github.com/therfoo/therfoo)](https://goreportcard.com/report/github.com/therfoo/therfoo)

# Therfoo
An easy to use machine learning Golang module.

## Example

```golang
package main

import (
	"github.com/therfoo/datasets/basic"

	"github.com/therfoo/therfoo/layers/dense"
	"github.com/therfoo/therfoo/model"
	"github.com/therfoo/therfoo/optimizers/sgd"
	"github.com/therfoo/therfoo/tensor"
)

func main() {
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
```
