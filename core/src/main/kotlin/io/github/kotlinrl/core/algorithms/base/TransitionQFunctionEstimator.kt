package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

interface TransitionQFunctionEstimator<State, Action> {
    fun estimate(Q: EnumerableQFunction<State, Action>, transition: Transition<State, Action>): EnumerableQFunction<State, Action>
}