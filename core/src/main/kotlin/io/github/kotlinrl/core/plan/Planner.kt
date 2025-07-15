package io.github.kotlinrl.core.plan

import io.github.kotlinrl.core.policy.Policy
import io.github.kotlinrl.core.policy.StateActionListProvider

interface Planner<State, Action> {
    fun plan(
        stateShape: IntArray,
        allActions: StateActionListProvider<State, Action>,
        transition: TransitionFunction<State, Action>,
        reward: RewardFunction<State, Action>,
    ): Policy<State, Action>
}