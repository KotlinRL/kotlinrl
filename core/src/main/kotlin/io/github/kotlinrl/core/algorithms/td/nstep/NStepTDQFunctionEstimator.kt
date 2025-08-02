package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.QFunction
import io.github.kotlinrl.core.Trajectory

interface NStepTDQFunctionEstimator<State, Action> {
    fun estimate(trajectory: Trajectory<State, Action>): QFunction<State, Action>
}