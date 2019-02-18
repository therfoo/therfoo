package metrics

import (
	"github.com/therfoo/therfoo/tensor"
)

func CategoricalAccuracy(yTrue, yEstimate *tensor.Vector) bool {
	yTrueMax, _ := yTrue.Max()
	yEstimateMax, _ := yEstimate.Max()
	return (yTrueMax == yEstimateMax)
}
