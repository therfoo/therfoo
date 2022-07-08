package layers

import (
	"math"
	"testing"
)

func TestConvolve(t *testing.T) {
	shape := Shape{5, 5}
	features := Random(shape.Size())

	l := &Conv{Filters: 1, FilterHeight: 3, FilterWidth: 3, Stride: 1}
	l.SetShape(shape)
	l.convolve(features)

	want := []float64{
		7.67101987,
		-6.21297594,
		-0.33452351,
		7.30678008,
		-9.25967344,
		6.99363256,
		-7.05250850,
		3.56739109,
		-3.16132225,
	}

	got := l.outputs[0]

	if len(want) != len(got) {
		t.Fatalf("want %d, got %d", len(want), len(got))
	}

	for k := range want {
		if math.Abs(got[k]-want[k]) > epsilon {
			t.Errorf("%d want %.8f, got %.8f", k, want[k], got[k])
		}
	}
}
