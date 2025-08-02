package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

interface MonteCarloValueFunctionEstimator<State, Action> {
    fun estimate(v: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State>
}
