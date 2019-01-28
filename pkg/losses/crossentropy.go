package losses

import (
	"github.com/therfoo/therfoo/pkg/tensor"
	"math"
)

func CrossEntropy(yTrue, yEstimate *tensor.Vector) *tensor.Vector {
	n := yTrue.Len()
	l := make(tensor.Vector, n, n)
	for i := 0; i < n; i++ {
		l[i] = -(*yTrue)[i]*math.Log((*yEstimate)[i]) - (1-(*yTrue)[i])*math.Log(1-(*yEstimate)[i])
	}
	return &l
}

func CrossEntropyPrime(yTrue, yEstimate *tensor.Vector) *tensor.Vector {
	n := yTrue.Len()
	d := make(tensor.Vector, n, n)
	for i := 0; i < n; i++ {
		d[i] = (*yEstimate)[i] - (*yTrue)[i]
	}
	return &d
}
