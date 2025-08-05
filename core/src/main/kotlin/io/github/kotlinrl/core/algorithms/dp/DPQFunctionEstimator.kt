package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

interface DPQFunctionEstimator<State, Action> {
    fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableQFunction<State, Action>
}
