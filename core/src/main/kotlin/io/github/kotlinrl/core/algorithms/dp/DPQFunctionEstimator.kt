package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

interface DPQFunctionEstimator<State, Action> {
    fun estimate(q:  QFunction<State, Action>, trajectory: ProbabilisticTrajectory<State, Action>): QFunction<State, Action>
}
