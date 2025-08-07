package io.github.kotlinrl.core.policy

import kotlin.math.*
import kotlin.random.*

/**
 * A policy implementation based on the softmax function, used in reinforcement learning
 * for stochastic action selection. The softmax policy assigns probabilities to actions
 * based on their Q-values, influenced by a temperature parameter. A lower temperature
 * increases the preference for actions with higher Q-values, making the policy more greedy,
 * while a higher temperature increases exploration by distributing probabilities more evenly
 * among the actions.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions available to the agent.
 * @param Q the Q-function mapping state-action pairs to their utility or quality values.
 * @param stateActions a function to determine the possible actions for a given state.
 * @param temperature a parameter schedule controlling the temperature, which affects
 *        the stochasticity of the policy. Lower values result in less exploration.
 * @param rng a random number generator for stochastic decision-making.
 */
class SoftmaxPolicy<State, Action>(
    override val Q: QFunction<State, Action>,
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
     * Creates an improved policy based on the provided Q-function by utilizing a softmax strategy.
     *
     * The resulting policy assigns probabilities to actions in each state determined by
     * their corresponding Q-values, influenced by a temperature parameter. This approach
     * facilitates exploration and balances the trade-off between exploitation and exploration.
     *
     * @param Q the Q-function representing the expected cumulative reward for each state-action pair.
     * @return the improved policy modeled as a softmax policy which uses the provided Q-function.
     */
    override fun improve(Q: QFunction<State, Action>): Policy<State, Action> =
        SoftmaxPolicy(Q, stateActions, temperature, rng)
}