package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

interface TrajectoryValueFunctionEstimator<State, Action> {
    fun estimate(V: EnumerableValueFunction<State>, trajectory: Trajectory<State, Action>): EnumerableValueFunction<State>
}
