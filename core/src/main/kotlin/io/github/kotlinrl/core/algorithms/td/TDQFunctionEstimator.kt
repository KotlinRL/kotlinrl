package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

interface TDQFunctionEstimator<State, Action> {
    fun estimate(q: QFunction<State, Action>, transition: Transition<State, Action>): QFunction<State, Action>
}