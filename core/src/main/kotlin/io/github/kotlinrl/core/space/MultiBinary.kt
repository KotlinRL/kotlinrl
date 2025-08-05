package io.github.kotlinrl.core.space

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * Represents a binary-valued space with a fixed size `n` where each element can independently take
 * the value 0 or 1. This class provides functionality to generate random samples from the space
 * and to check whether a given value is contained in the space.
 *
 * @property n The number of binary elements in the space.
 * @property seed An optional seed for the random number generator, enabling deterministic sampling.
 */
class MultiBinary(
    val n: Int,
    val seed: Int? = null
) : Space<NDArray<Int, D1>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Generates a random binary-valued NDArray of type Int and dimensionality D1.
     * Each element in the NDArray is independently assigned a value of either 0 or 1
     * with equal probability, based on the random number generator.
     *
     * @return An NDArray of type Int and dimensionality D1 with a size equal to `n`.
     *         The elements are randomly generated binary values (0 or 1).
     */
    override fun sample(): NDArray<Int, D1> =
        mk.ndarray(IntArray(n) { random.nextInt(2) }) // 0 or 1

    /**
     * Checks if the given value is contained within the space defined by the MultiBinary class.
     * The value is considered valid if it is an NDArray of type Int, has a dimensionality of D1,
     * a size equal to `n`, and each of its elements is either 0 or 1.
     *
     * @param value The value to check for containment. Expected to be an NDArray of type Int and dimensionality D1.
     * @return True if the value satisfies all the conditions of the binary-valued space, false otherwise.
     */
    override fun contains(value: Any?): Boolean =
        value is NDArray<*, *> &&
                value.dtype == DataType.IntDataType &&
                value.dim == D1 &&
                value.shape[0] == n &&
                value.indices.all { i ->
                    @Suppress("UNCHECKED_CAST")
                    val v = (value as NDArray<Int, D1>)[i]
                    v in 0..1
                }
}
