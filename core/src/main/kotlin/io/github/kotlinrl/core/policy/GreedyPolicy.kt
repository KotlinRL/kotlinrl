package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.EnumerableQFunction

class GreedyPolicy<State, Action>(
    override val Q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>
) : QFunctionPolicy<State, Action> {

    override operator fun invoke(state: State): Action {
        return Q.bestAction(state)
    }

    override fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        GreedyPolicy(Q, stateActions)
}