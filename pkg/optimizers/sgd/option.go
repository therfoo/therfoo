package sgd

type Option func(*SGD)

func WithBatchSize(batchSize int) Option {
	return func(o *SGD) {
		o.batchSize = batchSize
	}
}

func WithLearningRate(learningRate float64) Option {
	return func(o *SGD) {
		o.learningRate = learningRate
	}
}
