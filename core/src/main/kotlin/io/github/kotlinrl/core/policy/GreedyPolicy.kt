package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.algorithms.*

class GreedyPolicy<State, Action>(
    private val qTable: QFunction<State, Action>
) : Policy<State, Action> {

    override operator fun invoke(state: State): Action {
        return qTable.bestAction(state)
    }
}