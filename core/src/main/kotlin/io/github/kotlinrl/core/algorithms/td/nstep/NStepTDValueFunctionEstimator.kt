package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

interface NStepTDValueFunctionEstimator<State> {
    fun estimate(trajectory: Trajectory<State, *>): ValueFunction<State>
}
