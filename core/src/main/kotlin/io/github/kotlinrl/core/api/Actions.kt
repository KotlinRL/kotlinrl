package io.github.kotlinrl.core.api

/**
 * Represents a functional interface for defining actions associated with a specific state.
 * This can be used in scenarios such as decision-making systems or reinforcement learning,
 * where the actions available for a state need to be determined dynamically.
 *
 * @param State The type representing the state.
 * @param Action The type representing the action.
 */
fun interface Actions<State, Action> {
    /**
     * Retrieves the list of actions associated with a given state.
     *
     * This operator function is an alias for the `invoke` function, enabling convenient
     * syntax to fetch actions dynamically for a specified state.
     *
     * @param state The state for which the corresponding actions are to be determined.
     * @return A list of actions associated with the provided state.
     */
    operator fun get(state: State): Iterable<Action>
}