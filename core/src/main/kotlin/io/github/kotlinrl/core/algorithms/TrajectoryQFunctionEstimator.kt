package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

interface TrajectoryQFunctionEstimator<State, Action> {
    fun estimate(Q: EnumerableQFunction<State, Action>, trajectory: Trajectory<State, Action>): EnumerableQFunction<State, Action>
}