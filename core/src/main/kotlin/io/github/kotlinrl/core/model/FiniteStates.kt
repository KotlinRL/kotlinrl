package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

/**
 * Represents a finite set of integer states, commonly used in reinforcement learning environments
 * or other systems that require state enumeration and manipulation.
 *
 * @property numStates The total number of states in the finite set.
 * @property states A list of all states represented as integers, ranging from 0 to numStates - 1.
 * @property isFinite Indicates that the set of states is finite, always `true` for this class.
 * @property sizeOrNull The size of the state set, equal to numStates, as the set is finite.
 */
data class FiniteStates(val numStates: Int) : States<Int> {
    /**
     * Represents a sequentially ordered list of integer states, ranging from 0 to numStates - 1.
     *
     * This list is used within finite state representations to enumerate all possible states. It
     * provides a convenient way to access, iterate over, and check membership of state indices in
     * environments such as reinforcement learning or decision-making problems.
     *
     * @see iterator Used to iterate over the states in the list.
     * @see contains Checks if a specified state is present within the list.
     */
    val states: List<Int> = IntRange(0, numStates).toList()

    /**
     * Indicates that the set of states in this implementation is finite.
     *
     * This property is always set to `true` for instances of `FiniteStates`, as the class is specifically
     * designed to represent a finite collection of states. It helps distinguish finite state sets from
     * potentially infinite ones in interfaces and methods that handle generic state sets.
     */
    override val isFinite: Boolean = true

    /**
     * The size of the finite state set, equivalent to the total number of states represented.
     *
     * This property always returns a non-null value, as the state set is finite for this class. It
     * provides a direct way to determine the count of states without requiring additional computation.
     *
     * Overrides the `sizeOrNull` property from the `States` interface, ensuring it is non-null
     * for finite state sets.
     *
     * @return The exact number of states in the finite state set.
     */
    override val sizeOrNull: Int get() = states.size

    /**
     * Returns an iterator over the list of integers representing the states.
     * This allows sequential traversal of all states in the finite state set.
     *
     * @return An iterator of integers for the collection of states.
     */
    override fun iterator(): Iterator<Int> = states.iterator()

    /**
     * Checks if a given integer state is present in the finite set of states.
     *
     * @param state The integer state to check for membership within the set of states.
     * @return `true` if the specified state exists in the set; otherwise, `false`.
     */
    override fun contains(state: Int): Boolean = states.contains(state)
}