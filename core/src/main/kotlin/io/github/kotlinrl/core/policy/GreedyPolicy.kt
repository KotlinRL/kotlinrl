package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.EnumerableQFunction

/**
 * A deterministic policy implementation that always selects the action with the highest
 * Q-value for a given state. It is a greedy policy useful for exploiting the learned Q-function
 * to maximize reward.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param Q the Q-function used to evaluate the quality of state-action pairs.
 * @param stateActions a function providing the available actions for a given state.
 */
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