package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * Represents a stochastic policy implementation utilizing epsilon-soft action selection.
 * It ensures that the agent explores with probability epsilon while following the greedy policy
 * with probability (1 - epsilon). The exploration rate (epsilon) is determined dynamically
 * using the provided parameter schedule.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param Q the Q-function used to evaluate the quality of state-action pairs.
 * @param stateActions a function providing the available actions for a given state.
 * @param epsilon a parameter schedule that defines the value of epsilon, balancing exploration and exploitation.
 * @param rng the random number generator used for stochastic decisions.
 */
class EpsilonSoftPolicy<State, Action>(
    override val Q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val epsilon: ParameterSchedule,
    rng: Random
) : StochasticPolicy<State, Action>(rng) {

    /**
     * Computes the list of actions available in the given state along with their associated probabilities
     * based on the epsilon-greedy policy.
     *
     * @param state the current state for which the action probabilities are to be computed.
     * @return a list of pairs where each pair consists of an action and its associated probability.
     */
    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val actions = stateActions(state)
        val greedyAction = Q.bestAction(state)
        val n = actions.size
        val epsilon = epsilon()

        return actions.map { action ->
            val prob = if (action == greedyAction) {
                (1 - epsilon) + (epsilon / n)
            } else {
                epsilon / n
            }
            action to prob
        }
    }

    /**
     * Improves the given Q-function by returning an updated epsilon-soft policy.
     *
     * @param Q the Q-function to be improved, which provides state-action value information.
     * @return a new policy that implements an epsilon-soft approach using the provided Q-function.
     */
    override fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        EpsilonSoftPolicy(
            Q = Q,
            epsilon = epsilon,
            rng = rng,
            stateActions = stateActions
        )
}