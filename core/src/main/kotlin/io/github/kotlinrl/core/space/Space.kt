package io.github.kotlinrl.core.space

import kotlin.random.*

/**
 * Represents a generic space of type T, providing methods for sampling elements
 * and checking the membership of elements within the space.
 *
 * @param T The type of elements the space contains.
 */
interface Space<T> {
    val random: Random
    /**
     * Generates a random sample of type `T`.
     *
     * @return A randomly generated value of type `T`.
     */
    fun sample(): T
    /**
     * Checks if the specified value is contained within the space.
     *
     * @param value The value to check for membership in the space.
     *              It may need to conform to specific properties or constraints
     *              depending on the implementation of the containing class.
     * @return `true` if the provided value is considered a member of the space;
     *         `false` otherwise.
     */
    fun contains(value: Any?): Boolean
}