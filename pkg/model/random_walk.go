package model

import (
	"math/rand"
	"time"
)

func randomWalk(distance int) []int {
	rand.Seed(time.Now().Unix())
	return rand.Perm(distance)
}
