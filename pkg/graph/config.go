package graph

type Config struct {
	BatchSize    uint64
	LearningRate LearningRate
	Optimizer    Optimizer
	Regularizer  Regularizer
}

func (c Config) Validate() Config {
	c.BatchSize = max(1, c.BatchSize)
	if c.LearningRate == nil {
		c.LearningRate = DefaultLearningRate
	}
	return c
}
