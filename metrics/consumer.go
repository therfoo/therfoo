package metrics

type Consumer func(*Metrics) error
