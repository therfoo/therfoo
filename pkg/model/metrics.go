package model

type Metrics struct {
	Accuracy float64
	Cost     float64
}

type TrainingMetrics struct {
	Batch int
	Epoch int
	Metrics
}
