package graph

func argmax(x []float64) int {
	key := 0
	value := x[key]
	for k, v := range x {
		if v > value {
			key = k
			value = v
		}
	}
	return key
}
