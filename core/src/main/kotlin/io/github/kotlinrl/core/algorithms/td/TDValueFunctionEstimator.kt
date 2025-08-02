package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

interface TDValueFunctionEstimator<State, Action> {
    fun estimate(v: ValueFunction<State>, transition: Transition<State, Action>): ValueFunction<State>
}