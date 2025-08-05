package io.github.kotlinrl.core.space

import kotlin.random.*

/**
 * A class that represents a sequence space. Generates sequences of randomly sampled elements
 * from a given space with a bounded maximum length, and provides a mechanism to verify if a
 * given value belongs to the sequence space.
 *
 * @param T The type of elements in the sequence space.
 * @property space The space from which each element of the sequence is sampled.
 * @property maxLength The maximum possible length of the sequences in this space.
 * @property seed An optional seed for deterministic random number generation.
 */
class Sequence<T>(
    val space: Space<T>,
    val maxLength: Int,
    val seed: Int? = null
) : Space<List<T>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Generates a list of randomly sampled elements from the underlying space, with a length
     * determined randomly up to the specified maximum length (inclusive).
     *
     * @return A list of elements of type T, randomly sampled from the space. The length of the
     *         list is a random integer between 0 and the maximum length (inclusive).
     */
    override fun sample(): List<T> {
        val length = random.nextInt(maxLength + 1)
        return List(length) { space.sample() }
    }

    /**
     * Checks whether the given value is a valid sequence within the sequence space.
     * A value is considered valid if it is a list of elements, its size does not
     * exceed the defined maximum length, and all its elements are non-null and contained
     * within the associated space.
     *
     * @param value The value to check, expected to be of type `List<*>`.
     * @return `true` if the value is a list with a size less than or equal to the maximum
     *         length and all its elements are contained in the space; `false` otherwise.
     */
    override fun contains(value: Any?): Boolean =
        value is List<*> &&
                value.size <= maxLength &&
                value.all { it != null && space.contains(it) }
}
