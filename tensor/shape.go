package tensor

type Shape []int

func (s Shape) Size() int {
	params := 1
	for _, p := range s {
		params = params * p
	}
	return params
}
