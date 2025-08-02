package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

interface MDPModel<State, Action> {
    fun transitions(state: State, action: Action): ProbabilisticTrajectory<State, Action>

    fun allStates(): List<State>
}
