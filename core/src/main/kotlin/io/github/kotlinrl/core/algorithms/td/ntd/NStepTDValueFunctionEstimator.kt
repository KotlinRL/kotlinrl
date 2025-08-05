package io.github.kotlinrl.core.algorithms.td.ntd

import io.github.kotlinrl.core.*

interface NStepTDValueFunctionEstimator<State, Action> {
    fun estimate(v: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State>
}
