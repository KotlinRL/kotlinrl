package io.github.kotlinrl.core.policy

/**
 * Represents a specialized type of `ValueFunction` where all possible states
 * are enumerable. This allows explicit iteration over the set of all states
 * and facilitates value-based algorithms requiring comprehensive knowledge
 * of the state space.
 *
 * @param State the type representing the state in the environment.
 */
interface EnumerableValueFunction<State> : ValueFunction<State> {
    /**
     * Calculates and returns the maximum value available from the enumerable states.
     *
     * @return the maximum value among all enumerable states.
     */
    fun max(): Double

    /**
     * Retrieves all possible states in the environment. This function is useful
     * for contexts where the state space is enumerable, providing the ability
     * to iterate explicitly over all states.
     *
     * @return a list of all states within the environment.
     */
    fun allStates(): List<State>

    /**
     * Updates the value associated with a specific state and returns the updated enumerable value function.
     * This method modifies the state-value mapping in the current instance and results in a modified
     * enumerable value function reflecting the new value for the given state.
     *
     * @param state the state whose value is to be updated.
     * @param value the new value to associate with the specified state.
     * @return the updated instance of the enumerable value function.
     */
    override fun update(state: State, value: Double): EnumerableValueFunction<State>
}
