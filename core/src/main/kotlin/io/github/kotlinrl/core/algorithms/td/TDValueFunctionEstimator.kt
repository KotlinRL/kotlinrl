package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

interface TDValueFunctionEstimator<State> {
    fun estimate(v: ValueFunction<State>, transition: Transition<State, *>): ValueFunction<State>
}