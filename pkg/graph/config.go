package graph

type Config struct {
	BatchSize           uint64
	ClassWeights        []float64
	Data                Data
	DisableClassWeights bool
	DisableShuffle      bool
	Epochs              uint64
	LearningRate        LearningRate
	Optimizer           Optimizer
	Training            Data
	Validation          Data
	ValidationSplit     float64
}

func (c Config) Validate() Config {
	if c.ValidationSplit == 0 {
		c.Training = c.Data
	} else {
		c.Training, c.Validation = c.Data.Split(c.ValidationSplit)
	}
	if !c.DisableShuffle {
		c.Training.Shuffle()
	}
	if !c.DisableClassWeights {
		c.ClassWeights = c.Training.ClassWeights()
	}
	c.BatchSize = max(1, c.BatchSize)
	if c.LearningRate == nil {
		c.LearningRate = DefaultLearningRate
	}
	return c
}
