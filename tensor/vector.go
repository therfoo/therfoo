package tensor

type Vector []float64

func (v *Vector) Append(value float64) {
	*v = append(*v, value)
}

func (v *Vector) Each(f func(index int, value float64)) {
	for i := range *v {
		f(i, (*v)[i])
	}
}

func (v *Vector) Get(index int) float64 {
	return (*v)[index]
}

func (v *Vector) Len() int {
	return len(*v)
}
