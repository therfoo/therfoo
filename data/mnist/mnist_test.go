package mnist

import "testing"

func TestNew(t *testing.T) {
	tests := []struct {
		name       string
		batchCount int
		batchSize  int
		option     Option
	}{
		{"testing", 100, 100, WithTesting()},
		{"training", 600, 100, WithTraining()},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := New(WithBatchSize(test.batchSize), test.option)
			if g.Len() == test.batchCount {
				return
			}
			t.Errorf(
				"expected to find %d %s batches, found %d",
				test.batchCount,
				test.name,
				g.Len(),
			)
		})
	}
}
