package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.SoftmaxPolicy
import kotlin.math.*
import kotlin.random.*

/**
 * A policy that selects actions using the softmax function, commonly used in reinforcement learning
 * to model the probability distribution over actions based on their Q-values. The policy's behavior
 * is influenced by a temperature parameter, which controls the exploration versus exploitation trade-off.
 *
 * Higher temperature values lead to more exploration by making the action probabilities more uniform,
 * while lower values focus the probabilities around the best actions, promoting exploitation.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions available in the environment.
 * @param Q the Q-function that provides the expected cumulative rewards for each state-action pair.
 * @param stateActions a function that retrieves the list of available actions for a given state.
 * @param temperature a parameter schedule that determines the temperature value to be used in the softmax computation.
 * @param rng a random number generator used for stochastic sampling of actions.
 */
class SoftmaxPolicy<State, Action>(
    override val Q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val temperature: ParameterSchedule,
    rng: Random
) : StochasticPolicy<State, Action>(rng) {

    /**
     * Computes and returns a list of scored actions for the given state using the softmax function.
     *
     * The scores are calculated based on the Q-values of the state-action pairs and a temperature
     * parameter that influences the distribution of the scores. Higher scores indicate a higher
     * preference for the corresponding action.
     *
     * @param state the current state for which the action scores are to be computed.
     * @return a list of pairs where each pair consists of an action and its associated score.
     */
    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val temperature = temperature()
        val actions = stateActions(state)
        return actions.map { action ->
            action to exp(Q[state, action] / temperature)
        }
    }

    /**
     * Creates an improved policy based on the given Q-function using a softmax distribution.
     * The softmax function assigns probabilities to actions based on their Q-values and a
     * temperature parameter, allowing for stochastic decision-making. Higher Q-values result
     * in higher probabilities, and the temperature parameter influences the level of exploration.
     *
     * @param Q the Q-function that estimates the expected cumulative rewards for state-action pairs.
     * @return an improved policy represented as a softmax policy, which uses the given Q-function
     *         and softmax distribution for action selection.
     */
    override fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        SoftmaxPolicy(Q, stateActions, temperature, rng)
}