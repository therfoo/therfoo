package tensor

import (
	"testing"
)

func TestVectorAppend(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
	}{
		{"1", []float64{0.1, 0.1, 0.2, 0.3, 0.5, 0.8}},
		{"2", []float64{0.2, 0.1, 0.5, 0.3, 0.1, 0.5}},
		{"2", []float64{0.3, 0.2, 0.5, 0.1, 0.1, 0.3}},
		{"2", []float64{0.5, 0.3, 0.8, 0.2, 0.1, 0.1}},
		{"2", []float64{0.8, 0.5, 0.3, 0.5, 0.1, 0.2}},
	}
	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			v := Vector{}
			for j := range tests[i].values {
				v.Append(tests[i].values[j])
			}
			equal := func(a, b []float64) bool {
				if len(a) != len(b) {
					return false
				}
				for k := range a {
					if a[k] != b[k] {
						return false
					}
				}
				return true
			}(tests[i].values, v)
			if !equal {
				t.Errorf("expected vector to contain %v, found %v", tests[i].values, v)
			}
		})
	}
}

func TestVectorEach(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
	}{
		{"1", []float64{0.1, 0.1, 0.2, 0.3, 0.5, 0.8}},
		{"2", []float64{0.2, 0.1, 0.5, 0.3, 0.1, 0.5}},
		{"2", []float64{0.3, 0.2, 0.5, 0.1, 0.1, 0.3}},
		{"2", []float64{0.5, 0.3, 0.8, 0.2, 0.1, 0.1}},
		{"2", []float64{0.8, 0.5, 0.3, 0.5, 0.1, 0.2}},
	}
	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			v := Vector{}
			for j := range tests[i].values {
				v.Append(tests[i].values[j])
			}
			r := Vector{}
			v.Each(func(index int, value float64) {
				r.Append(value)
			})
			equal := func(a, b []float64) bool {
				if len(a) != len(b) {
					return false
				}
				for k := range a {
					if a[k] != b[k] {
						return false
					}
				}
				return true
			}(r, v)
			if !equal {
				t.Errorf("expected vector to contain %v, found %v", tests[i].values, v)
			}
		})
	}
}
