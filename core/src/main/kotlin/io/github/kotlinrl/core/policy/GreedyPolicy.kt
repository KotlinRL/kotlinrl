package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.EnumerableQFunction

class GreedyPolicy<State, Action>(
    override val q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>
) : QFunctionPolicy<State, Action> {

    override operator fun invoke(state: State): Action {
        return q.bestAction(state)
    }

    override fun improve(q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        GreedyPolicy(q, stateActions)
}