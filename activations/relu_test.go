package activations

import (
	"github.com/therfoo/therfoo/tensor"
	"testing"
)

func TestReLU(t *testing.T) {
	tests := []struct {
		name       string
		estimate   tensor.Scalar
		activation tensor.Scalar
	}{
		{"1", tensor.Scalar(-0.1), tensor.Scalar(0.)},
		{"2", tensor.Scalar(0.), tensor.Scalar(0.)},
		{"3", tensor.Scalar(0.2), tensor.Scalar(0.2)},
	}
	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			estimate := tests[i].estimate
			actual := ReLU(&estimate).(*tensor.Vector)
			if (*actual)[0] != float64(tests[i].activation) {
				t.Errorf("expected activation to be %v got %v", tests[i].activation, actual)
			}
		})
	}
}
