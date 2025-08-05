package io.github.kotlinrl.core.space

import kotlin.random.*

/**
 * Represents a discrete space with a finite number of integers, starting from a specified value.
 * This class allows sampling random integers within the defined range and checking whether a value
 * belongs to the space.
 *
 * @param n The number of discrete integers in the space.
 * @param start The starting value of the discrete range.
 * @param seed An optional seed for the random number generator. If null, the default random generator is used.
 */
class Discrete(
    val n: Int,
    val start: Int,
    val seed: Int? = null
) : Space<Int> {
    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Generates a random integer within the defined discrete space.
     * The range of the space is determined by the starting value and
     * the number of discrete integers (n).
     *
     * @return A random integer within the range [start, start + n).
     */
    override fun sample(): Int =
        start + random.nextInt(n)

    /**
     * Selects a random integer within the defined discrete space, optionally applying a mask
     * to limit the valid indices for sampling. If a mask is provided, only the indices marked
     * as `true` are considered valid for sampling.
     *
     * @param mask Optional BooleanArray where each element corresponds to an action. A `true`
     * value indicates the action is valid for sampling. The size of the mask must match the
     * number of discrete actions (n). If null, the method samples from the entire range.
     * @return A random integer within the defined range [start, start + n). If a mask is applied,
     * it selects a random integer corresponding to a valid index in the mask.
     * @throws IllegalArgumentException If the mask size does not match the number of discrete actions
     * (n) or if the mask does not allow at least one valid index.
     */
    fun sample(mask: BooleanArray? = null): Int =
        mask?.let {
            require(mask.size == n) { "Mask size must match the number of discrete actions (n)." }
            val validIndices = mask.withIndex().filter { it.value }.map { it.index }
            require(validIndices.isNotEmpty()) { "Mask must allow at least one valid action." }
            start + validIndices[random.nextInt(validIndices.size)]
        } ?: sample()

    /**
     * Checks if the given value is included within the discrete range defined by the
     * starting value (`start`) and the range length (`n`).
     *
     * @param value The value to check for inclusion. It can be any object, but only numerical
     *              values will be checked for containment. If the value is not a number, the
     *              method returns `false`.
     * @return `true` if the value falls within the range `[start, start + n)`, otherwise `false`.
     */
    override fun contains(value: Any?): Boolean =
        value in start until (start + n)

    /**
     * Returns a string representation of the `Discrete` class instance.
     * The string includes the values of the `start`, `n`, and `seed` properties.
     *
     * @return A string representation of the current object in the format "Discrete(start=<start>, n=<n>, seed=<seed>)".
     */
    override fun toString(): String {
        return "Discrete(start=$start, n=$n, seed=$seed)"
    }
}