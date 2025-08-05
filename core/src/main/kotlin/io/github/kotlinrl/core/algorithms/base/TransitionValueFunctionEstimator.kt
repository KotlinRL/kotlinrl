package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

interface TransitionValueFunctionEstimator<State, Action> {
    fun estimate(V: EnumerableValueFunction<State>, transition: Transition<State, Action>): EnumerableValueFunction<State>
}