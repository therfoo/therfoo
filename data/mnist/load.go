package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"github.com/therfoo/therfoo/tensor"
	"os"
)

const (
	dataDirectory     = "MNIST_DATA_DIR"
	imagesMagicNumber = 0x0803
	labelsMagicNumber = 0x0801
)

func Load(filename string) ([]tensor.Vector, error) {
	dir, exists := os.LookupEnv(dataDirectory)
	if !exists {
		return nil, fmt.Errorf("%s environment variable is not set.", dataDirectory)
	}
	file, err := os.Open(fmt.Sprintf("%s/%s", dir, filename))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}

	var magicNumber int32
	err = binary.Read(reader, binary.BigEndian, &magicNumber)
	if err != nil {
		return nil, err
	}

	var shape []int32

	switch magicNumber {
	case imagesMagicNumber:
		shape = []int32{1, 1, 1}
	case labelsMagicNumber:
		shape = []int32{1}
	default:
		err = fmt.Errorf("Invalid magic number (%d) detected.", magicNumber)
		return nil, err
	}

	var all int32 = 1
	for i := range shape {
		if err = binary.Read(reader, binary.BigEndian, &shape[i]); err != nil {
			return nil, err
		}
		all *= shape[i]
	}

	var size int32 = 1
	if magicNumber == imagesMagicNumber {
		size *= shape[1] * shape[2]
	} else {
		size = int32(1)
	}

	data := make([]tensor.Vector, shape[0], shape[0])

	var f uint8
	for i := int32(0); i < all; i++ {
		if err = binary.Read(reader, binary.BigEndian, &f); err != nil {
			return nil, err
		}
		j := i / size
		if magicNumber == imagesMagicNumber {
			data[j] = make(tensor.Vector, size, size)
		} else {

			data[j] = make(tensor.Vector, 10, 10)
		}
		if magicNumber == imagesMagicNumber {
			data[j][i%size] = float64(f) / 255.
		} else {
			data[j][f] = 1.
		}
	}

	return data, err
}
