package io.github.kotlinrl.core.policy

/**
 * Represents a value function in reinforcement learning, mapping states to scalar values
 * representing the estimated value of those states. This is commonly used in algorithms
 * to evaluate and compare states based on their expected future rewards.
 *
 * @param State the type representing the state in the environment.
 */
interface ValueFunction<State> {
    /**
     * Retrieves the estimated scalar value for the given state.
     *
     * This method is used to obtain the value associated with a specific state in the context
     * of a value function, which represents the expected future reward or value of that state.
     *
     * @param state the state for which to retrieve the estimated scalar value.
     * @return the scalar value representing the estimated value of the given state.
     */
    operator fun get(state: State): Double

    /**
     * Updates the value function with a new value for the specified state and returns the updated value function.
     *
     * This method modifies the value associated with the given state in the value function, reflecting
     * updated estimates for reinforcement learning purposes. The updated ValueFunction will include
     * the new value associated with the state.
     *
     * @param state the state for which the value should be updated.
     * @param value the new value to assign to the specified state.
     * @return the updated value function containing the new state-value mapping.
     */
    fun update(state: State, value: Double): ValueFunction<State>
}
