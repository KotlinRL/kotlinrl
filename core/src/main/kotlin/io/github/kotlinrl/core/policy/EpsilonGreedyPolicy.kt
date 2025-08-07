package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * A policy implementation that employs the epsilon-greedy strategy in reinforcement learning.
 *
 * The epsilon-greedy policy selects actions based on a trade-off between exploration and exploitation.
 * With a probability of epsilon, a random action is chosen (exploration), and with a probability of
 * 1 - epsilon, the action that maximizes the Q-value (greedy choice) is selected (exploitation).
 *
 * @param State the type representing the states of the environment.
 * @param Action the type representing the actions available in the environment.
 * @param Q the Q-function used to estimate the quality of state-action pairs and guide action selection.
 * @param stateActions a function that defines the set of possible actions available for each state.
 * @param epsilon a parameter schedule that determines the exploration rate.
 * @param rng the random number generator used to introduce randomness in the policy's exploration.
 */
class EpsilonGreedyPolicy<State, Action>(
    override val Q: QFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val epsilon: ParameterSchedule,
    private val rng: Random = Random.Default
) : Policy<State, Action> {
    private val randomPolicy = RandomPolicy(Q, stateActions, rng)
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
     * Improves the current policy based on the given Q-function.
     *
     * The improved policy is derived using the epsilon-greedy approach, where an action
     * is selected either greedily based on the Q-function or randomly with a probability
     * defined by the epsilon parameter.
     *
     * @param Q the Q-function representing the expected cumulative reward for each state-action pair.
     * @return the improved policy using the epsilon-greedy strategy for action selection.
     */
    override fun improve(Q: QFunction<State, Action>): Policy<State, Action> =
        EpsilonGreedyPolicy(
            Q = Q,
            stateActions = stateActions,
            epsilon = epsilon,
            rng = rng
        )
}