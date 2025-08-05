package io.github.kotlinrl.core.policy

/**
 * Represents a Q-function, a fundamental concept in reinforcement learning used to estimate the
 * quality of state-action pairs. The "quality" corresponds to the expected cumulative reward
 * when starting from a specific state and taking a particular action.
 *
 * This interface defines the contract for a generic Q-function, allowing implementation and
 * manipulation of state-action value estimations.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
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
}
