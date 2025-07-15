package io.github.kotlinrl.core.plan

import io.github.kotlinrl.core.*

interface Planner<State, Action> {
    fun plan(
        stateActionListProvider: StateActionListProvider<State, Action>,
        transitionFunction: TransitionFunction<State, Action>
    ): Policy<State, Action>
}