package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * Implements an epsilon-greedy policy for reinforcement learning.
 * The epsilon-greedy policy selects a random action with a probability
 * given by the epsilon parameter, and selects the greedy (highest Q-value) action otherwise.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the action that can be performed in the environment.
 * @property Q the Q-function used to evaluate state-action value pairs.
 * @property stateActions a function that provides the list of available actions for a given state.
 * @property epsilon a schedule defining the epsilon value, which controls the trade-off
 * between exploration and exploitation.
 * @property rng the random number generator used to select random actions.
 */
class EpsilonGreedyPolicy<State, Action>(
    override val Q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val epsilon: ParameterSchedule,
    private val rng: Random = Random.Default
) : QFunctionPolicy<State, Action> {
    private val randomPolicy = RandomPolicy(stateActions, rng)
    private val greedyPolicy = GreedyPolicy(Q, stateActions)

    /**
     * Invokes the epsilon-greedy policy to select an action based on the current state.
     * The action is chosen either through exploration (random action) or exploitation (greedy action),
     * depending on the epsilon value.
     *
     * @param state the current state of the environment.
     * @return the action to be taken, selected according to the epsilon-greedy strategy.
     */
    override fun invoke(state: State): Action =
        if (rng.nextDouble() < epsilon()) {
            randomPolicy(state)
        } else {
            greedyPolicy(state)
        }

    /**
     * Creates and returns an updated policy based on the given Q-function using the epsilon-greedy strategy.
     * The epsilon-greedy policy balances exploration and exploitation by selecting a random action with
     * probability epsilon, and the best action as per the Q-function with probability (1 - epsilon).
     *
     * @param Q the Q-function to generate the improved policy. It represents the expected rewards for
     *          each state-action pair and helps guide the action selection process.
     * @return an epsilon-greedy policy that leverages the updated Q-function to make decisions.
     */
    override fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        EpsilonGreedyPolicy(
            Q = Q,
            stateActions = stateActions,
            epsilon = epsilon,
            rng = rng
        )
}