package io.github.kotlinrl.core.policy

/**
 * Represents a Q-function in reinforcement learning, mapping state-action pairs to scalar values
 * representing the estimated utility or quality of those pairs. Q-functions are used to evaluate
 * and improve policies by quantifying the expected cumulative reward of taking specific actions
 * in specific states.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions available to the agent in the environment.
 */
interface QFunction<State, Action> {
    /**
     * Retrieves the Q-value associated with a given state-action pair.
     *
     * @param state the state for which the Q-value is to be determined.
     * @param action the action corresponding to the given state for which the Q-value is to be retrieved.
     * @return the Q-value representing the quality of the specified state-action pair.
     */
    operator fun get(state: State, action: Action): Double

    /**
     * Updates the Q-value associated with a specific state-action pair and returns the updated Q-function.
     *
     * @param state the state for which the Q-value should be updated.
     * @param action the action associated with the state to update the Q-value.
     * @param value the new Q-value to be assigned to the specified state-action pair.
     * @return the updated Q-function with the modified Q-value for the given state-action pair.
     */
    fun update(state: State, action: Action, value: Double): QFunction<State, Action>

    /**
     * Computes the maximum Q-value for a given state.
     *
     * @param state the state for which the maximum Q-value is to be calculated.
     * @return the maximum Q-value among all possible actions for the given state.
     */
    fun maxValue(state: State): Double

    /**
     * Determines the best action to take for a given state based on the Q-function.
     *
     * @param state the current state for which the best action is to be determined.
     * @return the action that maximizes the Q-value for the given state.
     */
    fun bestAction(state: State): Action

    /**
     * Converts the Q-function into a value function by deriving the maximum Q-value for each state.
     *
     * The resulting value function represents the estimated value of each state, calculated as the maximum
     * Q-value over all possible actions for that state within the context of the Q-function.
     *
     * @return a value function mapping states to their maximum Q-values, representing the optimal value of each state.
     */
    fun toV(): ValueFunction<State>
}
