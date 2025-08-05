package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.policy.EnumerableValueFunction

interface DPValueFunctionEstimator<State, Action> {
    fun estimate(
        V: EnumerableValueFunction<State>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableValueFunction<State>
}
