package io.github.kotlinrl.core.api

/**
 * Represents a collection of states in a reinforcement learning environment or similar systems.
 *
 * This interface defines common properties and operations for handling states, including the ability
 * to check if the set of states is finite, retrieve the size of the set, and check for membership of a specific state.
 *
 * @param State The type representing the states in the environment.
 */
interface States<State> : Iterable<State> {
    /**
     * Indicates whether the set of states represented by this instance is finite.
     *
     * A finite state set implies that the total number of states is countable and known,
     * while an infinite or unknown state set would return `false` for this property.
     *
     * This property is beneficial in determining the feasibility of exhaustive operations
     * over the state space, such as enumeration or certain evaluations, which might not
     * be possible for an infinite set of states.
     */
    val isFinite: Boolean
    /**
     * Represents the size of the state set if it is finite, or `null` if the size is infinite or unknown.
     *
     * This property is particularly useful for distinguishing between state sets that are explicitly countable
     * and those that cannot be quantified due to their infinite nature or lack of clear bounds.
     *
     * When `isFinite` is `true`, this property provides the exact count of states.
     * When `isFinite` is `false`, the property will always return `null` to indicate indeterminacy.
     */
    val sizeOrNull: Int?        // null if infinite/unknown
    /**
     * Checks if the specified state is present in the set of states.
     *
     * @param state the state to check for membership in the set of states.
     * @return `true` if the state is part of the set, otherwise `false`.
     */
    fun contains(state: State): Boolean
}