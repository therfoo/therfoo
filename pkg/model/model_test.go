package model

import (
	"fmt"
	"github.com/therfoo/therfoo/pkg/layers/dense"
	"github.com/therfoo/therfoo/pkg/optimizers/sgd"
	"github.com/therfoo/therfoo/pkg/tensor"
	"math"
	"testing"
)

var testData = []struct {
	features tensor.Vector
	labels   tensor.Vector
}{
	{tensor.Vector{0.78051, -0.063669}, tensor.Vector{1}},
	{tensor.Vector{0.28774, 0.29139}, tensor.Vector{1}},
	{tensor.Vector{0.40714, 0.17878}, tensor.Vector{1}},
	{tensor.Vector{0.2923, 0.4217}, tensor.Vector{1}},
	{tensor.Vector{0.50922, 0.35256}, tensor.Vector{1}},
	{tensor.Vector{0.27785, 0.10802}, tensor.Vector{1}},
	{tensor.Vector{0.27527, 0.33223}, tensor.Vector{1}},
	{tensor.Vector{0.43999, 0.31245}, tensor.Vector{1}},
	{tensor.Vector{0.33557, 0.42984}, tensor.Vector{1}},
	{tensor.Vector{0.23448, 0.24986}, tensor.Vector{1}},
	{tensor.Vector{0.0084492, 0.13658}, tensor.Vector{1}},
	{tensor.Vector{0.12419, 0.33595}, tensor.Vector{1}},
	{tensor.Vector{0.25644, 0.42624}, tensor.Vector{1}},
	{tensor.Vector{0.4591, 0.40426}, tensor.Vector{1}},
	{tensor.Vector{0.44547, 0.45117}, tensor.Vector{1}},
	{tensor.Vector{0.42218, 0.20118}, tensor.Vector{1}},
	{tensor.Vector{0.49563, 0.21445}, tensor.Vector{1}},
	{tensor.Vector{0.30848, 0.24306}, tensor.Vector{1}},
	{tensor.Vector{0.39707, 0.44438}, tensor.Vector{1}},
	{tensor.Vector{0.32945, 0.39217}, tensor.Vector{1}},
	{tensor.Vector{0.40739, 0.40271}, tensor.Vector{1}},
	{tensor.Vector{0.3106, 0.50702}, tensor.Vector{1}},
	{tensor.Vector{0.49638, 0.45384}, tensor.Vector{1}},
	{tensor.Vector{0.10073, 0.32053}, tensor.Vector{1}},
	{tensor.Vector{0.69907, 0.37307}, tensor.Vector{1}},
	{tensor.Vector{0.29767, 0.69648}, tensor.Vector{1}},
	{tensor.Vector{0.15099, 0.57341}, tensor.Vector{1}},
	{tensor.Vector{0.16427, 0.27759}, tensor.Vector{1}},
	{tensor.Vector{0.33259, 0.055964}, tensor.Vector{1}},
	{tensor.Vector{0.53741, 0.28637}, tensor.Vector{1}},
	{tensor.Vector{0.19503, 0.36879}, tensor.Vector{1}},
	{tensor.Vector{0.40278, 0.035148}, tensor.Vector{1}},
	{tensor.Vector{0.21296, 0.55169}, tensor.Vector{1}},
	{tensor.Vector{0.48447, 0.56991}, tensor.Vector{1}},
	{tensor.Vector{0.25476, 0.34596}, tensor.Vector{1}},
	{tensor.Vector{0.21726, 0.28641}, tensor.Vector{1}},
	{tensor.Vector{0.67078, 0.46538}, tensor.Vector{1}},
	{tensor.Vector{0.3815, 0.4622}, tensor.Vector{1}},
	{tensor.Vector{0.53838, 0.32774}, tensor.Vector{1}},
	{tensor.Vector{0.4849, 0.26071}, tensor.Vector{1}},
	{tensor.Vector{0.37095, 0.38809}, tensor.Vector{1}},
	{tensor.Vector{0.54527, 0.63911}, tensor.Vector{1}},
	{tensor.Vector{0.32149, 0.12007}, tensor.Vector{1}},
	{tensor.Vector{0.42216, 0.61666}, tensor.Vector{1}},
	{tensor.Vector{0.10194, 0.060408}, tensor.Vector{1}},
	{tensor.Vector{0.15254, 0.2168}, tensor.Vector{1}},
	{tensor.Vector{0.45558, 0.43769}, tensor.Vector{1}},
	{tensor.Vector{0.28488, 0.52142}, tensor.Vector{1}},
	{tensor.Vector{0.27633, 0.21264}, tensor.Vector{1}},
	{tensor.Vector{0.39748, 0.31902}, tensor.Vector{1}},
	{tensor.Vector{0.5533, 1}, tensor.Vector{0}},
	{tensor.Vector{0.44274, 0.59205}, tensor.Vector{0}},
	{tensor.Vector{0.85176, 0.6612}, tensor.Vector{0}},
	{tensor.Vector{0.60436, 0.86605}, tensor.Vector{0}},
	{tensor.Vector{0.68243, 0.48301}, tensor.Vector{0}},
	{tensor.Vector{1, 0.76815}, tensor.Vector{0}},
	{tensor.Vector{0.72989, 0.8107}, tensor.Vector{0}},
	{tensor.Vector{0.67377, 0.77975}, tensor.Vector{0}},
	{tensor.Vector{0.78761, 0.58177}, tensor.Vector{0}},
	{tensor.Vector{0.71442, 0.7668}, tensor.Vector{0}},
	{tensor.Vector{0.49379, 0.54226}, tensor.Vector{0}},
	{tensor.Vector{0.78974, 0.74233}, tensor.Vector{0}},
	{tensor.Vector{0.67905, 0.60921}, tensor.Vector{0}},
	{tensor.Vector{0.6642, 0.72519}, tensor.Vector{0}},
	{tensor.Vector{0.79396, 0.56789}, tensor.Vector{0}},
	{tensor.Vector{0.70758, 0.76022}, tensor.Vector{0}},
	{tensor.Vector{0.59421, 0.61857}, tensor.Vector{0}},
	{tensor.Vector{0.49364, 0.56224}, tensor.Vector{0}},
	{tensor.Vector{0.77707, 0.35025}, tensor.Vector{0}},
	{tensor.Vector{0.79785, 0.76921}, tensor.Vector{0}},
	{tensor.Vector{0.70876, 0.96764}, tensor.Vector{0}},
	{tensor.Vector{0.69176, 0.60865}, tensor.Vector{0}},
	{tensor.Vector{0.66408, 0.92075}, tensor.Vector{0}},
	{tensor.Vector{0.65973, 0.66666}, tensor.Vector{0}},
	{tensor.Vector{0.64574, 0.56845}, tensor.Vector{0}},
	{tensor.Vector{0.89639, 0.7085}, tensor.Vector{0}},
	{tensor.Vector{0.85476, 0.63167}, tensor.Vector{0}},
	{tensor.Vector{0.62091, 0.80424}, tensor.Vector{0}},
	{tensor.Vector{0.79057, 0.56108}, tensor.Vector{0}},
	{tensor.Vector{0.58935, 0.71582}, tensor.Vector{0}},
	{tensor.Vector{0.56846, 0.7406}, tensor.Vector{0}},
	{tensor.Vector{0.65912, 0.71548}, tensor.Vector{0}},
	{tensor.Vector{0.70938, 0.74041}, tensor.Vector{0}},
	{tensor.Vector{0.59154, 0.62927}, tensor.Vector{0}},
	{tensor.Vector{0.45829, 0.4641}, tensor.Vector{0}},
	{tensor.Vector{0.79982, 0.74847}, tensor.Vector{0}},
	{tensor.Vector{0.60974, 0.54757}, tensor.Vector{0}},
	{tensor.Vector{0.68127, 0.86985}, tensor.Vector{0}},
	{tensor.Vector{0.76694, 0.64736}, tensor.Vector{0}},
	{tensor.Vector{0.69048, 0.83058}, tensor.Vector{0}},
	{tensor.Vector{0.68122, 0.96541}, tensor.Vector{0}},
	{tensor.Vector{0.73229, 0.64245}, tensor.Vector{0}},
	{tensor.Vector{0.76145, 0.60138}, tensor.Vector{0}},
	{tensor.Vector{0.58985, 0.86955}, tensor.Vector{0}},
	{tensor.Vector{0.73145, 0.74516}, tensor.Vector{0}},
	{tensor.Vector{0.77029, 0.7014}, tensor.Vector{0}},
	{tensor.Vector{0.73156, 0.71782}, tensor.Vector{0}},
	{tensor.Vector{0.44556, 0.57991}, tensor.Vector{0}},
	{tensor.Vector{0.85275, 0.85987}, tensor.Vector{0}},
	{tensor.Vector{0.51912, 0.62359}, tensor.Vector{0}},
}

type TestGenerator struct{}

func (g TestGenerator) Next(index int) (x, y *[]tensor.Vector) {
	data := testData[index]
	return &[]tensor.Vector{data.features}, &[]tensor.Vector{data.labels}
}

func (g TestGenerator) Len() int {
	return len(testData)
}

func TestFit(t *testing.T) {
	lr := 0.005
	m := New(
		WithEpochs(1000),
		WithInputShape([]int{2}),
		WithOptimizer(
			sgd.New(
				sgd.WithBatchSize(1),
				sgd.WithLearningRate(lr),
			),
		),
		WithCrossEntropyLoss(),
		WithTestingGenerator(TestGenerator{}),
		WithTrainingGenerator(TestGenerator{}),
		WithValidatingGenerator(TestGenerator{}),
	)

	m.Add(8, dense.New(dense.WithReLU()))
	m.Add(1, dense.New(dense.WithSigmoid()))

	m.Compile()

	trainingMetrics := make(chan *TrainingMetrics, 10)

	var evaluate = func(msg string, n *Model) {
		accurate, total := 0., 0.

		for i := range testData {
			y := m.Predict(&[]tensor.Vector{testData[i].features})
			if math.Round((*y)[0][0]) == testData[i].labels[0] {
				accurate++
			}
			total++
		}

		fmt.Printf("%s: %.0f%%, LR: %f\n", msg, (accurate/total)*100., lr)
	}

	go m.Fit(trainingMetrics)

	epoch := 0
	for m := range trainingMetrics {
		if m.Epoch > epoch {
			epoch = m.Epoch
		}
	}
	evaluate("Training accuracy", m)
}
