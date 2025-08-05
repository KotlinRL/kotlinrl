package io.github.kotlinrl.core.policy

/**
 * Represents a specialized version of the `QFunction` interface where
 * all possible states are enumerable. This allows explicit iteration
 * over the entire state space and facilitates algorithms requiring
 * complete knowledge of the environment's structure.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the action that can be taken in the environment.
 */
interface EnumerableQFunction<State, Action> : QFunction<State, Action> {
    /**
     * Retrieves a list of all possible states in the environment. This method is designed for use
     * cases where the state space is enumerable and provides a comprehensive view of all states.
     *
     * @return a list containing all states within the environment.
     */
    fun allStates(): List<State>

    /**
     * Updates the Q-value associated with a specific state-action pair to the specified value and
     * returns the resulting updated Q-function. This method modifies the current Q-function to
     * reflect the new value for the given state-action pair while leaving other entries unchanged.
     *
     * @param state the state for which the Q-value needs to be updated.
     * @param action the action associated with the specified state.
     * @param value the value to be assigned to the state-action pair.
     * @return the updated Q-function reflecting the change to the specified state-action pair.
     */
    override fun update(
        state: State,
        action: Action,
        value: Double
    ): EnumerableQFunction<State, Action>;
}
