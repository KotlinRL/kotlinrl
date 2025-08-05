package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

interface DPQFunctionEstimator<State, Action> {
    fun estimate(
        q: EnumerableQFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableQFunction<State, Action>
}
