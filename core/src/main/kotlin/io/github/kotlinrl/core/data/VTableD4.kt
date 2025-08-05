package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * A specialized implementation of `EnumerableValueFunction` for 4-dimensional state spaces,
 * represented as `NDArray<Int, D3>`. This class encapsulates a value table for states with
 * exactly 4 dimensions and provides operations to retrieve, update, and save state values.
 *
 * @constructor
 * Initializes the instance with the specified shape. The shape must include exactly 4 dimensions.
 * Throws an `IllegalArgumentException` if the shape does not contain exactly 4 arguments.
 *
 * @property shape The dimensions of the 4-dimensional state space.
 */
class VTableD4(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D3>> {

    init {
        require(shape.size == 4) { "VTableD4 shape requires exactly 4 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    /**
     * Retrieves the value associated with a given 4-dimensional state represented as `NDArray<Int, D3>`.
     *
     * @param state The 4-dimensional state to retrieve the value for. Must be an instance of `NDArray<Int, D3>`.
     * @return The value associated with the provided state as a `Double`.
     */
    override fun get(state: NDArray<Int, D3>): Double =
        base[state.asDNArray()]

    /**
     * Updates the value associated with a given 4-dimensional state represented as `NDArray<Int, D3>`.
     * The updated value is stored in the internal table of the copied instance of the VTableD4 object.
     *
     * @param state The 4-dimensional state to be updated. Must be an instance of `NDArray<Int, D3>`.
     * @param value The new value to associate with the provided state.
     * @return A new instance of `EnumerableValueFunction<NDArray<Int, D3>>` with the updated value.
     */
    override fun update(state: NDArray<Int, D3>, value: Double): EnumerableValueFunction<NDArray<Int, D3>> =
        copy().also { it.base.table[state.toIntArray()] = value }

    /**
     * Retrieves all possible 3-dimensional states represented as `NDArray<Int, D3>`.
     * Converts the list of raw states from the base implementation to a list of 3-dimensional arrays.
     *
     * @return A list of 3-dimensional states, where each state is an instance of `NDArray<Int, D3>`.
     */
    override fun allStates(): List<NDArray<Int, D3>> =
        base.allStates().map { it.asD3Array() }

    /**
     * Computes the maximum value stored in the internal table of this `VTableD4` instance.
     *
     * @return The maximum value as a `Double`.
     */
    override fun max(): Double =
        base.max()

    /**
     * Creates a new copy of the current `VTableD4` instance.
     * The new instance will have the same shape and identical data as the original.
     *
     * @return A new `VTableD4` instance with copied internal data.
     */
    fun copy(): VTableD4 =
        VTableD4(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the current internal table of the `VTableD4` instance to a CSV file at the specified path.
     * This operation ensures that the data is persisted in a format that can be retrieved or processed later.
     *
     * @param path The file path where the internal table will be saved as a CSV file.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads data from a CSV file located at the specified path and reshapes it
     * into the multi-dimensional array structure corresponding to the current object's shape.
     * The reshaped data is then copied into the internal table of this `VTableD4` instance.
     *
     * @param path The file path where the CSV file is located. The file should contain data
     *             that can be reshaped into the dimensions defined by this object's shape.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the internal table of the `VTableD4` instance to the standard output.
     * The table represents the multi-dimensional data associated with the object.
     */
    fun print() = base.print()

    /**
     * Converts the current `VTableD4` instance into a `VTableD5` instance with the specified shape.
     * The method initializes a new `VTableD5` object, copies the data from the current table into the new table,
     * and returns the created instance.
     *
     * @param shape The dimensions for the new VTableD5. The provided arguments must form a valid 4-dimensional shape.
     * @return A new instance of `VTableD5` with the specified shape and copied data from the current instance.
     */
    fun asVTable5(vararg shape: Int): VTableD5 =
        VTableD5(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current `VTableD4` instance into a `VTableDN` instance with the specified shape.
     * The method initializes a new `VTableDN` object, copies the data from the current table into the new table,
     * and returns the created instance.
     *
     * @param shape Variable number of integers representing the desired dimensions for the new `VTableDN`.
     *              Must contain at least two dimensions.
     * @return A new `VTableDN` instance with the specified shape and copied data from the current instance.
     */
    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}