package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

interface NStepTDValueFunctionEstimator<State, Action> {
    fun estimate(V: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State>
}
