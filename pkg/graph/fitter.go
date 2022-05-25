package graph

type Fitter struct {
	Epochs               uint64
	Training, Validation Features
}

func (f Fitter) Prepare() Fitter {
	f.Training.Prepare()
	f.Validation.Prepare()
	return f
}

func (f Fitter) Fit(g Graph, w ...MetricsWriter) {
	for i := 1; i <= int(f.Epochs); i++ {
		for j := range f.Training.X {
			a := g.Estimate(f.Training.X[j])
			gradients := make([]float64, len(a))
			for k := range gradients {
				gradients[k] = a[k] - f.Training.Y[j][k]
				if f.Training.ClassWeights != nil {
					gradients[k] *= f.Training.ClassWeights[k]
				}
			}
			g.Minimize(gradients)
			for k := range w {
				w[k].Write(Metrics{Epoch: i, Sample: j, Estimate: a, Actual: f.Training.Y[j]})
			}
		}
		for j := range f.Validation.X {
			a := g.Estimate(f.Validation.X[j])
			gradients := make([]float64, len(a))
			for k := range gradients {
				gradients[k] = a[k] - f.Validation.Y[j][k]
			}
			for k := range w {
				w[k].Write(Metrics{Epoch: i, Sample: j, Estimate: a, Actual: f.Validation.Y[j]})
			}
		}
	}
}
