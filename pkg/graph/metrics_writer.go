package graph

type MetricsWriter interface {
	Write(Metrics)
}

type MetricsWriterFunc func(Metrics)

func (fn MetricsWriterFunc) Write(m Metrics) {
	fn(m)
}
