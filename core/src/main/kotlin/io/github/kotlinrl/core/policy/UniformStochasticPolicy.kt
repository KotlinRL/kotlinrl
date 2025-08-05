package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * A stochastic policy that selects actions uniformly at random from the set of available actions
 * for a given state. This policy is based on a random uniform distribution and uses the available
 * action space to ensure each action is chosen with equal probability.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param Q the Q-function associated with this policy, which is not directly used in decision-making
 *          but required for compatibility and improvement methods.
 * @param stateActions a function providing the set of available actions for each state.
 * @param rng an optional random number generator for stochastic action sampling.
 */
class UniformStochasticPolicy<State, Action>(
    override val Q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    rng: Random = Random.Default
) : StochasticPolicy<State, Action>(rng) {
    private val randomPolicy = RandomPolicy(stateActions, rng)

    /**
     * Invokes the policy to select an action based on the given state using a random policy.
     *
     * @param state the current state of the environment for which an action is to be selected.
     * @return the selected action for the given state.
     */
    override fun invoke(state: State): Action {
        return randomPolicy(state)
    }

    /**
     * Creates and returns an improved version of the policy based on the given Q-function.
     *
     * The improvement entails constructing a new `UniformStochasticPolicy` that continues to select
     * actions uniformly at random but now utilizes the updated Q-function to maintain compatibility
     * with the reinforcement learning framework.
     *
     * @param Q the Q-function representing the expected cumulative reward for each state-action pair.
     * @return the improved policy, represented as a `UniformStochasticPolicy` based on the provided Q-function.
     */
    override fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        UniformStochasticPolicy(Q, stateActions, rng)

    /**
     * Computes the scores for all possible actions in a given state, assuming a uniform probability
     * distribution across all actions. Each action is assigned an equal probability.
     *
     * @param state the current state of the environment for which the action scores are to be computed.
     * @return a list of pairs where each pair consists of an action and its corresponding score (probability).
     */
    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val actions = stateActions(state)
        val p = 1.0 / actions.size
        return actions.map { it to p }
    }
}
