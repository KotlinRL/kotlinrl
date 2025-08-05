package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

interface TransitionQFunctionEstimator<State, Action> {
    fun estimate(q: EnumerableQFunction<State, Action>, transition: Transition<State, Action>): EnumerableQFunction<State, Action>
}