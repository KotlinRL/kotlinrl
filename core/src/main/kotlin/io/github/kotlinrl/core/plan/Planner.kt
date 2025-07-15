package io.github.kotlinrl.core.plan

import io.github.kotlinrl.core.*

interface Planner<State, Action> {
    fun plan(
        stateShape: IntArray,
        stateActionListProvider: StateActionListProvider<State, Action>,
        transitionFunction: TransitionFunction<State, Action>,
        rewardFunction: RewardFunction<State, Action>,
    ): Policy<State, Action>
}