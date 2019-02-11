package model

import (
	"github.com/therfoo/therfoo/optimizers"
	"github.com/therfoo/therfoo/tensor"
	"math"
	"math/rand"
	"sync"
	"time"
)

type Model struct {
	lossFunction func(yTrue, yEstimate *tensor.Vector) *tensor.Vector
	lossPrime    func(yTrue, yEstimate *tensor.Vector) *tensor.Vector
	epochs       int
	generator    struct {
		testing    Generator
		training   Generator
		validating Generator
	}
	inputShape           []int
	layers               []Layer
	layersCount          int
	learningRate         float64
	neuronsCount         []int
	optimizer            optimizers.Optimizer
	skipOutputDerivative bool
}

func (m *Model) activate(activation *tensor.Vector) *[]tensor.Vector {
	n := m.layersCount + 1
	activations := make([]tensor.Vector, n, n)

	activations[0] = *activation

	for l := 0; l < m.layersCount; l++ {
		activations[l+1] = m.layers[l].Activate(&(activations[l]))
	}

	return &activations
}

func (m *Model) Add(neuronsCount int, layer Layer) {
	m.neuronsCount = append(m.neuronsCount, neuronsCount)
	m.layers = append(m.layers, layer)
	m.layersCount++
}

func (m *Model) inputSize() int {
	params := 1
	for _, c := range m.inputShape {
		params = params * c
	}
	return params
}

func (m *Model) Compile() error {
	neurons := make([][]int, m.layersCount, m.layersCount)

	for l := range m.layers {
		var weightsCount int
		if l == 0 {
			weightsCount = m.inputSize()
		} else {
			weightsCount = m.neuronsCount[l-1]
		}
		m.layers[l].Init(m.neuronsCount[l], weightsCount)
		neurons[l] = []int{m.neuronsCount[l], weightsCount}
	}

	m.optimizer.Init(&neurons)

	return nil
}

func (m *Model) minimize(yTrue *tensor.Vector, activations *[]tensor.Vector) {

	yEstimate := (*activations)[m.layersCount]
	nablaLoss := m.lossPrime(yTrue, &yEstimate)

	if !m.skipOutputDerivative {
		m.layers[m.layersCount-1].Derive(&((*activations)[m.layersCount-1]), nablaLoss)
	}
	m.optimizer.Add(m.layersCount-1, &((*activations)[m.layersCount-1]), nablaLoss)

	ll := m.layersCount - 2
	for l := ll; l > -1; l-- {
		nablaLoss = m.layers[l+1].NextCost(nablaLoss)
		m.layers[l].Derive(&((*activations)[l+1]), nablaLoss)
		m.optimizer.Add(l, &((*activations)[l]), nablaLoss)
	}
}

func (m *Model) evaluate(generator Generator, c chan *Metrics) {
	rand.Seed(time.Now().Unix())
	steps := rand.Perm(generator.Len())
	for _, step := range steps {
		xBatch, yBatch := generator.Get(step)
		c <- m.evaluateBatch(xBatch, yBatch)
	}
	close(c)
}

func (m *Model) evaluateBatch(xBatch, yBatch *[]tensor.Vector) *Metrics {
	metrics := Metrics{}
	yBatchEstimate := m.Predict(xBatch)
	// TODO:
	accurate, total := 0., 0.
	for i := range *yBatchEstimate {
		yTrue := (*yBatch)[i]
		yEstimate := (*yBatchEstimate)[i]
		// TODO:
		yTrue.Each(func(index int, value float64) {
			if value == math.Round(yEstimate[index]) {
				accurate++
			}
		})
		total++
	}
	metrics.Accuracy = (accurate / total) * 100.
	metrics.Cost = metrics.Cost / float64(len(*xBatch))
	return &metrics
}

func (m *Model) optimize(optimizations *[][][]float64) {
	for layer := range *optimizations {
		m.layers[layer].Adjust(&((*optimizations)[layer]))
	}
}

func (m *Model) Predict(xBatch *[]tensor.Vector) *[]tensor.Vector {
	var predictions []tensor.Vector
	for i := range *xBatch {
		activations := m.activate(&(*xBatch)[i])
		predictions = append(predictions, (*activations)[m.layersCount-1])
	}
	return &predictions
}

func (m *Model) Test(c chan *Metrics) {
	m.evaluate(m.generator.testing, c)
}

func (m *Model) train(xBatch, yBatch *[]tensor.Vector) {

	wg := sync.WaitGroup{}

	batchSize := len(*xBatch)

	wg.Add(batchSize)

	for i := range *xBatch {
		go func(s int) {
			x := (*xBatch)[s]
			m.minimize(&((*yBatch)[s]), m.activate(&x))
			wg.Done()
		}(i)
	}

	wg.Wait()

	m.optimize(m.optimizer.Optimizations())

}

func (m *Model) Fit(metrics chan *TrainingMetrics) {
	n := m.generator.training.Len()
	for e := 0; e < m.epochs; e++ {
		rand.Seed(time.Now().Unix())
		for batch, step := range rand.Perm(n) {
			xBatch, yBatch := m.generator.training.Get(step)
			m.train(xBatch, yBatch)
			metrics <- &TrainingMetrics{
				Metrics: *m.evaluateBatch(xBatch, yBatch),
				Epoch:   e,
				Batch:   batch,
			}
		}
	}
	close(metrics)
}

func (m *Model) Validate(c chan *Metrics) {
	m.evaluate(m.generator.validating, c)
}

func New(options ...Option) *Model {
	m := Model{}
	for i := range options {
		options[i](&m)
	}
	return &m
}
