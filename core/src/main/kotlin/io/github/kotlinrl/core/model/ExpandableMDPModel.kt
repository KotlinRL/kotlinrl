package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

/**
 * Interface representing an expandable Markov Decision Process (MDP) model.
 *
 * This interface defines methods for managing and querying the model state,
 * including state predecessors, visit counts, and terminal state identification.
 *
 * @param State The type representing states in the MDP model.
 * @param Action The type representing actions in the MDP model.
 */
interface ExpandableMDPModel<State, Action> {
    /**
     * Retrieves the set of predecessor state-action pairs for a given state.
     *
     * A predecessor is defined as a state-action pair where executing the
     * action in the state leads to the specified state.
     *
     * @param state The target state for which to find the predecessor state-action pairs.
     * @return A set of `StateActionKey` objects representing the predecessor state-action pairs
     *         that can transition to the given state.
     */
    fun predecessors(state: State): Set<StateActionKey<*, *>>
    /**
     * Retrieves the visit count for a specific state-action pair in the Markov Decision Process (MDP).
     *
     * @param state The state in the MDP for which the visit count is being queried.
     * @param action The action associated with the state in the MDP for which the visit count is being queried.
     * @return The number of times the specified state-action pair has been visited.
     */
    fun visitCount(state: State, action: Action): Int
    /**
     * Determines if the given state is a terminal state in the Markov Decision Process (MDP) model.
     *
     * A terminal state is defined as a state where no further transitions are possible
     * or where the MDP process ends.
     *
     * @param state The state to be checked for terminality.
     * @return True if the state is terminal, false otherwise.
     */
    fun isTerminal(state: State): Boolean
}
