package layers

import (
	"math"
	"testing"
)

func TestConvolve2D(t *testing.T) {
	// Original 2D test scenario
	shape := Shape{5, 5}
	features := Random(shape.Size())

	l := &Conv{Filters: 1, FilterHeight: 3, FilterWidth: 3, Stride: 1}
	l.SetShape(shape.Shape())
	l.convolve(features)

	// The expected values from the original code snippet
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

	got := l.output
	if len(want) != len(got) {
		t.Fatalf("2D: want %d, got %d", len(want), len(got))
	}

	for k := range want {
		if math.Abs(got[k]-want[k]) > epsilon {
			t.Errorf("2D: %d want %.8f, got %.8f", k, want[k], got[k])
		}
	}
}

func TestConvolve1D(t *testing.T) {
	// 1D convolution scenario
	// Input shape: [10]
	// FilterWidth: 3, Stride: 1 -> output width = 10 - 3 + 1 = 8
	// Output shape = [8, Filters], with Filters=1
	shape := Shape{10}
	features := Random(shape.Size())

	l := &Conv{Filters: 1, FilterWidth: 3, Stride: 1}
	// For 1D, FilterHeight and FilterDepth default to 1
	l.SetShape(shape.Shape())
	l.convolve(features)

	got := l.output

	// Length should be 8 * Filters = 8
	if len(got) != 8 {
		t.Fatalf("1D: expected output length of 8, got %d", len(got))
	}

	// If you want exact numerical tests, run once, print got, then paste here:
	// Example (placeholders, you should replace after a real run):
	want := []float64{
		0.12345678,
		-0.23456789,
		1.23456789,
		-1.34567890,
		0.98765432,
		-0.87654321,
		0.11111111,
		-0.22222222,
	}

	if len(want) != len(got) {
		t.Fatalf("1D: want length %d, got %d", len(want), len(got))
	}

	for i := range want {
		if math.Abs(got[i]-want[i]) > epsilon {
			t.Errorf("1D: index %d, want %.8f, got %.8f", i, want[i], got[i])
		}
	}
}

func TestConvolve3D(t *testing.T) {
	// 3D convolution scenario
	// Input shape: [D, H, W] = [2, 5, 5]
	// Filters=1, FilterDepth=2, FilterHeight=3, FilterWidth=3, Stride=1
	// Output depth = 2 - 2 + 1 = 1
	// Output height = 5 - 3 + 1 = 3
	// Output width = 5 - 3 + 1 = 3
	// Output shape = [1, 3, 3, 1] = total 1*3*3*1 = 9 elements
	shape := Shape{2, 5, 5}
	features := Random(shape.Size())

	l := &Conv{Filters: 1, FilterDepth: 2, FilterHeight: 3, FilterWidth: 3, Stride: 1}
	l.SetShape(shape.Shape())
	t.Logf("inputShape: %v", l.inputShape)
	l.convolve(features)

	got := l.output

	// Expecting 9 output values
	if len(got) != 9 {
		t.Fatalf("3D: want length 9, got %d", len(got))
	}

	// After running once, capture output and store as want:
	want := []float64{
		0.01234567,
		-0.12345678,
		0.23456789,
		-0.34567890,
		0.45678901,
		-0.56789012,
		0.67890123,
		-0.78901234,
		0.89012345,
	}

	if len(want) != len(got) {
		t.Fatalf("3D: want length %d, got %d", len(want), len(got))
	}

	for i := range want {
		if math.Abs(got[i]-want[i]) > epsilon {
			t.Errorf("3D: index %d, want %.8f, got %.8f", i, want[i], got[i])
		}
	}
}
