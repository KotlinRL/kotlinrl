package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * Represents an epsilon-soft policy in reinforcement learning, which introduces stochasticity
 * into the policy selection process. This policy promotes exploration by occasionally choosing
 * suboptimal actions with a probability controlled by the epsilon parameter, while still favoring
 * the optimal action as determined by the Q-function.
 *
 * This approach ensures a balance between exploration and exploitation, making it suitable for various
 * reinforcement learning tasks where an agent needs to learn an optimal policy over time.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions available in the environment.
 * @param Q the Q-function used to estimate the value of state-action pairs.
 * @param stateActions a functional interface that provides the list of possible actions for a given state.
 * @param epsilon a parameter schedule that determines the exploration rate at any given time.
 * @param rng a random number generator used for stochastic behavior in action selection.
 */
class EpsilonSoftPolicy<State, Action>(
    override val Q: QFunction<State, Action>,
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
     * Improves the current policy by generating an epsilon-soft policy using the given Q-function.
     *
     * The epsilon-soft policy ensures that actions are chosen based on an epsilon-greedy strategy,
     * where the agent usually selects the action with the highest Q-value, but occasionally explores
     * other actions with a probability defined by epsilon.
     *
     * @param Q the Q-function representing the expected cumulative reward for each state-action pair.
     * @return an improved policy that utilizes the provided Q-function for decision-making.
     */
    override fun improve(Q: QFunction<State, Action>): Policy<State, Action> =
        EpsilonSoftPolicy(
            Q = Q,
            epsilon = epsilon,
            rng = rng,
            stateActions = stateActions
        )
}