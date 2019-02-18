package metrics

import (
	"github.com/therfoo/therfoo/tensor"
)

func BinaryAccuracy(yTrue, yEstimate *tensor.Vector) bool {
	return ((*yTrue)[0] == 1 && (*yEstimate)[0] > 0.5) || ((*yTrue)[0] == 0 && (*yEstimate)[0] <= 0.5)
}
