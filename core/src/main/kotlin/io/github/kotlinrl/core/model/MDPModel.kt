package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

interface MDPModel<State, Action> {
    fun allStates(): List<State>

    fun allActions(): List<Action>

    fun transitions(state: State, action: Action): ProbabilisticTrajectory<State, Action>

    fun expectedReward(state: State, action: Action): Double
}
