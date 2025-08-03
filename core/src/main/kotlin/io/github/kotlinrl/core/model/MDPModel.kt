package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

interface MDPModel<State, Action> {
    fun transitions(state: State, action: Action): ProbabilisticTrajectory<State, Action>

    fun allStates(): List<State>
}
