package layers

import "github.com/therfoo/therfoo/pkg/graph"

var AND = graph.Features{
	X: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
	Y: [][]float64{{0}, {0}, {0}, {1}},
}

var NAND = graph.Features{
	X: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
	Y: [][]float64{{1}, {1}, {1}, {0}},
}

var OR = graph.Features{
	X: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
	Y: [][]float64{{0}, {1}, {1}, {1}},
}

var XOR = graph.Features{
	X: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
	Y: [][]float64{{0}, {1}, {1}, {0}},
}
