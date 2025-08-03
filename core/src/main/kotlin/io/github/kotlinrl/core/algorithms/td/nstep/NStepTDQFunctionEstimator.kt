package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

interface NStepTDQFunctionEstimator<State, Action> {
    fun estimate(q: QFunction<State, Action>, trajectory: Trajectory<State, Action>): QFunction<State, Action>
}