package metrics

import (
	"fmt"
)

func Logger(m *Metrics) error {
	_, err := fmt.Printf(
		"Epoch: %d, Accuracy: %.4f Cost: %.4f\n",
		m.Epoch+1,
		m.Accuracy,
		m.Cost,
	)
	return err
}
