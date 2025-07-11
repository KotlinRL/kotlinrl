package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable

class GreedyPolicy<State, Action>(
    private val actionProvider: StateActionListProvider<State, Action>,
    private val qTable: QTable<State, Action>
) : Policy<State, Action> {

    override operator fun invoke(state: State): Action {
        val availableActions = actionProvider(state)
        require(availableActions.isNotEmpty()) { "No available actions for state: $state" }
        return availableActions.maxByOrNull { qTable[state, it] }
            ?: error("No max action found. Ensure stateActionRewardProvider works correctly.")
    }
}