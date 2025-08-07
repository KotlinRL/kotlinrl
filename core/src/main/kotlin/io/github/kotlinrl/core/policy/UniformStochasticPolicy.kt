package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * A uniform stochastic policy for reinforcement learning, which selects actions uniformly at random
 * from the set of possible actions for a given state.
 *
 * This policy leverages a random policy internally and is designed to facilitate exploration in
 * reinforcement learning environments by ensuring an equal probability distribution across all
 * feasible actions for a state.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions available to the agent.
 * @param Q the Q-function used for evaluating state-action pairs.
 * @param stateActions a functional interface providing the set of feasible actions for a given state.
 * @param rng the random number generator used to introduce stochasticity in action selection.
 */
class UniformStochasticPolicy<State, Action>(
    override val Q: QFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    rng: Random = Random.Default
) : StochasticPolicy<State, Action>(rng) {
    private val randomPolicy = RandomPolicy(Q, stateActions, rng)

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
     * Creates an improved policy based on the provided Q-function. The improved policy uses a uniform
     * stochastic strategy to optimize decision-making, ensuring that actions are selected stochastically
     * based on the given state's available actions and the Q-function-defined quality values.
     *
     * @param Q the Q-function mapping state-action pairs to their estimated utility or quality values.
     *          It is used to evaluate the expected reward for each state-action combination.
     * @return a new policy instance, based on a uniform stochastic strategy, that optimizes action selection
     *         by leveraging the provided Q-function and supported state-action configurations.
     */
    override fun improve(Q: QFunction<State, Action>): Policy<State, Action> =
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
