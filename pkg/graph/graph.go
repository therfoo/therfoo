package graph

import "math"

type Graph []Layer

func (g Graph) Apply(c Config) {
	for i := range g {
		if layer, ok := g[i].(*Minimizer); ok {
			layer.learningRate = c.LearningRate
			layer.optimizer = c.Optimizer
			layer.regularizer = c.Regularizer
			layer.batch = make([][][]float64, c.BatchSize)
			for j := range layer.batch {
				layer.batch[j] = make([][]float64, len(layer.gradients))
				for k := range layer.gradients {
					layer.batch[j][k] = make([]float64, len(layer.gradients[k]))
				}
			}
		}
	}
}

func (g Graph) Estimate(x []float64) []float64 {
	for i := range g {
		x = g[i].Estimate(x)
	}
	return x
}

func (g Graph) Fit(c Config, w ...MetricsWriter) {
	for i := 1; i <= int(c.Epochs); i++ {
		for j := range c.Training.X {
			a := g.Estimate(c.Training.X[j])
			gradients := make([]float64, len(a))
			for k := range gradients {
				gradients[k] = a[k] - c.Training.Y[j][k]
				if c.ClassWeights != nil {
					gradients[k] *= c.ClassWeights[k]
				}
			}
			g.Minimize(gradients)
			for k := range w {
				w[k].Write(Metrics{Epoch: i, Sample: j, Estimate: a, Actual: c.Training.Y[j]})
			}
		}
		for j := range c.Validation.X {
			a := g.Estimate(c.Validation.X[j])
			gradients := make([]float64, len(a))
			for k := range gradients {
				gradients[k] = a[k] - c.Validation.Y[j][k]
			}
			for k := range w {
				w[k].Write(Metrics{Epoch: i, Sample: j, Estimate: a, Actual: c.Validation.Y[j]})
			}
		}
	}
}

func (g Graph) Gradients() [][][]float64 {
	gradients := make([][][]float64, len(g))
	for i := range gradients {
		if l, ok := g[i].(Minimizeable); ok {
			gradients[i] = l.Gradients()
		}
	}
	return gradients
}

func (g Graph) Loss(x, y []float64) []float64 {
	a := g.Estimate(x)
	loss := make([]float64, len(y))
	for i := range loss {
		loss[i] = a[i] - y[i]
	}
	return loss
}

func (g Graph) Minimize(gradients []float64) []float64 {
	j := len(g) - 1
	for i := range g {
		gradients = g[j-i].Minimize(gradients)
	}
	return gradients
}

func (g Graph) NumericGradients(x, y []float64) [][][]float64 {
	var cost = func(a, y []float64) float64 {
		var sum float64
		for i := range a {
			sum += math.Pow(a[i]-y[i], 2)
		}
		return 0.5 * sum
	}

	var zeros = func(a [][][]float64) [][][]float64 {
		b := make([][][]float64, len(a))
		for i := range a {
			b[i] = make([][]float64, len(a[i]))
			for j := range a[i] {
				b[i][j] = make([]float64, len(a[i][j]))
			}
		}
		return b
	}

	weights := g.Weights()
	gradients := zeros(weights)
	for i := range weights {
		for j := range weights[i] {
			for k := range weights[i][j] {
				w := weights[i][j][k]

				weights[i][j][k] += epsilon
				plus := cost(g.Estimate(x), y)
				weights[i][j][k] = w

				weights[i][j][k] -= epsilon
				minus := cost(g.Estimate(x), y)
				weights[i][j][k] = w

				gradients[i][j][k] = (plus - minus) / (2 * epsilon)
			}
		}
	}
	return gradients
}

func (g Graph) Weights() [][][]float64 {
	weights := make([][][]float64, len(g))
	for i := range g {
		if learner, ok := g[i].(Minimizeable); ok {
			weights[i] = learner.Weights()
		}
	}
	return weights
}

func New(layers ...Layer) Graph {
	var graph = Graph(layers)
	for i := 1; i < len(layers); i++ {
		graph[i].SetShape(layers[i-1].Shape())
	}
	for i := range graph {
		if layer, ok := graph[i].(Minimizeable); ok {
			minimizer := Minimizer{gradients: layer.Gradients(), weights: layer.Weights()}
			minimizer.Layer = graph[i]
			graph[i] = &minimizer
		}
	}
	return graph
}
