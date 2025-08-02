package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

interface MonteCarloQFunctionEstimator<State, Action> {
    fun estimate(q: QFunction<State, Action>, trajectory: Trajectory<State, Action>, episode: Int): QFunction<State, Action>
}
