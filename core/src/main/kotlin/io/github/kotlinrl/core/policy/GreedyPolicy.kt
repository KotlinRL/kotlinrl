package io.github.kotlinrl.core.policy

class GreedyPolicy<State, Action>(
    override val qTable: QFunction<State, Action>
) : QFunctionPolicy<State, Action> {

    override operator fun invoke(state: State): Action {
        return qTable.bestAction(state)
    }
}