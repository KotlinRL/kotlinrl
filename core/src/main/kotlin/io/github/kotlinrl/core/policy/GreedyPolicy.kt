package io.github.kotlinrl.core.policy

class GreedyPolicy<State, Action>(
    override val q: QFunction<State, Action>
) : QFunctionPolicy<State, Action>, PolicyImprovementStrategy<State, Action> {

    override operator fun invoke(state: State): Action {
        return q.bestAction(state)
    }

    override operator fun invoke(q: QFunction<State, Action>): Policy<State, Action> =
        GreedyPolicy(q)
}