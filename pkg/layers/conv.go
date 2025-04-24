package layers

import (
	"math"
)

type Conv struct {
	layer
	learner
	Filters, FilterDepth, FilterHeight, FilterWidth, Stride int
	// The dimensions after convolution
	outputDepth, outputHeight, outputWidth int
}

func (l *Conv) convolve(features []float64) {
	// Determine input dimension
	var D, H, W int
	switch len(l.inputShape) {
	case 1:
		// 1D: shape = [W]
		W = l.inputShape.Width()
		H = 1
		D = 1
	case 2:
		// 2D: shape = [H, W]
		H = l.inputShape.Height()
		W = l.inputShape.Width()
		D = 1
	case 3:
		// 3D: shape = [D, H, W]
		D = int(l.inputShape[0])
		H = int(l.inputShape[1])
		W = int(l.inputShape[2])
	default:
		panic("Conv layer only supports 1D, 2D, or 3D input shapes")
	}

	for f := 0; f < l.Filters; f++ {
		for od := 0; od < l.outputDepth; od += l.Stride {
			for oh := 0; oh < l.outputHeight; oh += l.Stride {
				for ow := 0; ow < l.outputWidth; ow += l.Stride {
					outputIndex := l.indexOutput(f, od, oh, ow)
					var sum float64
					for kd := 0; kd < l.FilterDepth; kd++ {
						for kh := 0; kh < l.FilterHeight; kh++ {
							for kw := 0; kw < l.FilterWidth; kw++ {
								inD := od + kd
								inH := oh + kh
								inW := ow + kw
								if inD < D && inH < H && inW < W {
									inputIndex := l.indexInput(D, H, W, inD, inH, inW)
									filterIndex := kd*l.FilterHeight*l.FilterWidth + kh*l.FilterWidth + kw
									sum = math.FMA(features[inputIndex], l.weights[f][filterIndex], sum)
								}
							}
						}
					}
					l.output[outputIndex] = sum
				}
			}
		}
	}
}

func (l *Conv) Estimate(features []float64) []float64 {
	l.convolve(features)
	return l.output
}

func (l *Conv) SetShape(shape []uint64) {
	l.inputShape = Shape(shape)

	// Determine input dimension
	var D, H, W int
	switch len(shape) {
	case 1:
		// 1D: shape = [W]
		W = int(shape[0])
		H = 1
		D = 1
		// For 1D conv, FilterDepth and FilterHeight should be 1
		if l.FilterDepth == 0 {
			l.FilterDepth = 1
		}
		if l.FilterHeight == 0 {
			l.FilterHeight = 1
		}
	case 2:
		// 2D: shape = [H, W]
		H = int(shape[0])
		W = int(shape[1])
		D = 1
		// For 2D conv, FilterDepth should be 1
		if l.FilterDepth == 0 {
			l.FilterDepth = 1
		}
	case 3:
		// 3D: shape = [D, H, W]
		D = int(shape[0])
		H = int(shape[1])
		W = int(shape[2])
	default:
		panic("Conv layer only supports 1D, 2D, or 3D input shapes")
	}

	// Calculate output dimensions
	l.outputDepth = D - l.FilterDepth + 1
	l.outputHeight = H - l.FilterHeight + 1
	l.outputWidth = W - l.FilterWidth + 1

	// Validate output dimensions
	if l.outputDepth <= 0 || l.outputHeight <= 0 || l.outputWidth <= 0 {
		panic("Invalid filter size for given input shape")
	}

	// Set final output shape
	var outShape []uint64
	switch len(shape) {
	case 1:
		// 1D output: [outputWidth, Filters]
		outShape = []uint64{uint64(l.outputWidth), uint64(l.Filters)}
	case 2:
		// 2D output: [outputHeight, outputWidth, Filters]
		outShape = []uint64{uint64(l.outputHeight), uint64(l.outputWidth), uint64(l.Filters)}
	case 3:
		// 3D output: [outputDepth, outputHeight, outputWidth, Filters]
		outShape = []uint64{uint64(l.outputDepth), uint64(l.outputHeight), uint64(l.outputWidth), uint64(l.Filters)}
	}
	l.layer.SetOutputShape(outShape)

	// Initialize weights
	filterSize := l.FilterDepth * l.FilterHeight * l.FilterWidth
	l.weights = make([][]float64, l.Filters)
	l.localGradients = make([][]float64, l.Filters)
	for i := 0; i < l.Filters; i++ {
		l.weights[i] = Random(filterSize)
		l.localGradients[i] = make([]float64, filterSize)
	}
}

func (l *Conv) Shape() []uint64 {
	return l.outputShape.Shape()
}

func (l *Conv) Minimize(gradients []float64) []float64 {
	// Backprop not implemented here; just return input for now.
	return gradients
}

// indexInput converts 3D indices (for 1D/2D/3D) into a single index
func (l *Conv) indexInput(D, H, W, d, h, w int) int {
	// For 1D: d=0, h=0 and w varies
	// For 2D: d=0, h and w vary
	// For 3D: d, h, w vary
	return d*H*W + h*W + w
}

// indexOutput converts 3D output indices into a single index
// Output is stored as (Filters, D', H', W') or (Filters, H', W') or (W', Filters) depending on dimension.
func (l *Conv) indexOutput(f, d, h, w int) int {
	switch len(l.outputShape) {
	case 2:
		// 1D: output shape: [outputWidth, Filters]
		// index: w*Filters + f
		return w*int(l.Filters) + f
	case 3:
		// 2D: output shape: [outputHeight, outputWidth, Filters]
		// index: (h*outputWidth + w)*Filters + f
		return (h*l.outputWidth+w)*l.Filters + f
	case 4:
		// 3D: output shape: [outputDepth, outputHeight, outputWidth, Filters]
		// index: ((d*outputHeight + h)*outputWidth + w)*Filters + f
		return ((d*l.outputHeight+h)*l.outputWidth+w)*l.Filters + f
	default:
		panic("Unsupported output shape dimensions")
	}
}
