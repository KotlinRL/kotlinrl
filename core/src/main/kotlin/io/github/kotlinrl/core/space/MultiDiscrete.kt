package io.github.kotlinrl.core.space

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * Represents a space consisting of multiple discrete dimensions, where each dimension has a finite range
 * of integers defined by the respective values in the `nvec` array. This class supports sampling random
 * values within the specified ranges and checking if a given value is valid within this space.
 *
 * @constructor Initializes a MultiDiscrete space with the given dimension sizes and an optional random seed.
 * Each element of `nvec` specifies the size of the corresponding dimension.
 *
 * @param nvec A variable number of integers representing the size of each discrete dimension.
 * @param seed An optional integer seed for the random number generator, enabling reproducible random sampling.
 */
class MultiDiscrete(
    vararg val nvec: Int,
    val seed: Int? = null
) : Space<NDArray<Int, D1>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Generates a random sample within the discrete space defined by the `nvec` array.
     * Each dimension in the resulting array corresponds to the defined range in `nvec`,
     * with its value randomly sampled from 0 (inclusive) to the maximum value for that dimension (exclusive).
     *
     * @return An NDArray of integers with one dimension (`D1`), where each element is a random
     *         value within the range specified by the corresponding index in `nvec`.
     */
    override fun sample(): NDArray<Int, D1> =
        mk.ndarray(IntArray(nvec.size) { random.nextInt(nvec[it]) })

    /**
     * Checks if the given value is a valid instance of `NDArray<Int, D1>` that conforms to the
     * constraints defined by the `nvec` array in this discrete space.
     *
     * Each element of the `NDArray` must correspond to an integer within the range [0, nvec[i])
     * for the respective dimension `i`.
     *
     * @param value The object to check for validity within the discrete space. This must be an
     *              instance of `NDArray<Int, D1>` with dimensions and values matching the criteria.
     * @return `true` if the value is a valid `NDArray` that satisfies the space's constraints;
     *         otherwise, `false`.
     */
    override fun contains(value: Any?): Boolean =
        value is NDArray<*, *> &&
                value.dtype == DataType.IntDataType &&
                value.dim == D1 &&
                value.shape[0] == nvec.size &&
                @Suppress("UNCHECKED_CAST")
                (value as NDArray<Int, D1>).indices.all {
                    value[it] in 0 until nvec[it]
                }
}
