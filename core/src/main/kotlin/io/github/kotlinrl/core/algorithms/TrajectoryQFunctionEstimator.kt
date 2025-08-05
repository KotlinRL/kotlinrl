package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

interface TrajectoryQFunctionEstimator<State, Action> {
    fun estimate(q: EnumerableQFunction<State, Action>, trajectory: Trajectory<State, Action>): EnumerableQFunction<State, Action>
}