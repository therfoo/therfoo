package mnist

import "os"

type Option func(*MNIST)

func WithBatchSize(size int) Option {
	return func(m *MNIST) {
		m.batchSize = size
	}
}

func WithTraining() Option {
	return func(m *MNIST) {
		var err error
		if m.images, err = Load("train-images-idx3-ubyte.gz"); err != nil {
			print(err.Error())
			print("\n")
			os.Exit(1)
		}
		if m.labels, err = Load("train-labels-idx1-ubyte.gz"); err != nil {
			print(err.Error())
			print("\n")
			os.Exit(1)
		}
	}
}

func WithTesting() Option {
	return func(m *MNIST) {
		var err error
		if m.images, err = Load("t10k-images-idx3-ubyte.gz"); err != nil {
			print(err.Error())
			print("\n")
			os.Exit(1)
		}
		if m.labels, err = Load("t10k-labels-idx1-ubyte.gz"); err != nil {
			print(err.Error())
			print("\n")
			os.Exit(1)
		}
	}
}
