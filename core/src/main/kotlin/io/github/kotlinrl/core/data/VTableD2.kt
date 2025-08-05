package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * Represents a two-dimensional implementation of an enumerable value function.
 * This class allows storing and managing state-value pairs for an environment
 * where the state space has a fixed two-dimensional shape.
 *
 * @constructor Initializes a new instance with the specified shape.
 * The shape must consist of exactly two dimensions.
 *
 * @param shape The dimensions for the two-dimensional state space.
 *
 * @throws IllegalArgumentException If the shape does not contain exactly two dimensions.
 */
class VTableD2(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D1>> {

    init {
        require(shape.size == 2) { "VTableD2 shape requires exactly 2 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    /**
     * Retrieves a `Double` value corresponding to the specified state in the VTable.
     *
     * @param state The state represented as an NDArray of integers with one dimension (D1).
     * @return The value associated with the given state as a Double.
     */
    override fun get(state: NDArray<Int, D1>): Double =
        base[state.asDNArray()]

    /**
     * Updates the VTable with a new value for the specified state.
     *
     * @param state The state represented as an NDArray of integers with one dimension (D1).
     * @param value The new value to associate with the given state.
     * @return A new instance of EnumerableValueFunction with the updated state-value mapping.
     */
    override fun update(state: NDArray<Int, D1>, value: Double): EnumerableValueFunction<NDArray<Int, D1>> =
        copy().also { it.base.table[state.toIntArray()] = value }


    /**
     * Retrieves all possible states represented as one-dimensional NDArray objects (D1).
     *
     * @return A list of NDArray objects, where each NDArray represents a possible state in the VTable with one dimension (D1).
     */
    override fun allStates(): List<NDArray<Int, D1>> =
        base.allStates().map { it.asD1Array() }

    /**
     * Finds and returns the maximum value present in the `base` data structure.
     *
     * @return The maximum value as a Double.
     */
    override fun max(): Double =
        base.max()

    /**
     * Creates a copy of the current `VTableD2` instance.
     *
     * @return A new `VTableD2` instance with the same structure and data as the original.
     */
    fun copy(): VTableD2 =
        VTableD2(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the current VTable data to a specified file path in CSV format.
     *
     * @param path The file path where the VTable data will be saved. The data will be stored
     *             in CSV format, ensuring compatibility with external tools and systems.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads data from a CSV file located at the specified file path and populates the internal
     * data structure of the VTable. The method reshapes the data to match the specified dimensions
     * of the VTable based on its shape and updates the table contents accordingly.
     *
     * @param path The file path of the CSV file to be loaded. The file must contain numeric data
     *             formatted in a way that can be reshaped to fit the dimensions of the VTable.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the contents of the `base` field of the VTableD2 instance to the standard output.
     *
     * The method utilizes the `print()` function from the `base` object to handle the actual printing.
     * Typically used for debugging or inspecting the internal state of the VTable.
     */
    fun print() = base.print()

    /**
     * Converts the current VTableD2 instance into a new VTableD3 instance with the specified shape.
     * Copies the internal data from the current VTableD2 instance into the created VTableD3 instance.
     *
     * @param shape The shape of the new VTableD3 instance. Must contain exactly 3 integer dimensions.
     * @return A new VTableD3 instance with the specified shape, containing data copied from the current instance.
     * @throws IllegalArgumentException If the provided shape does not have exactly 3 dimensions.
     */
    fun asVTable3(vararg shape: Int): VTableD3 =
        VTableD3(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current instance into a new `VTableD4` with the specified shape.
     * Copies the internal data from the current instance into the new `VTableD4` instance.
     *
     * @param shape The shape of the new `VTableD4` instance. Must contain exactly 4 integer dimensions.
     * @return A new `VTableD4` instance with the specified shape, containing data copied from the current instance.
     * @throws IllegalArgumentException If the provided shape does not have exactly 4 dimensions.
     */
    fun asVTable4(vararg shape: Int): VTableD4 =
        VTableD4(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current instance into a new `VTableD5` with the specified shape.
     * Copies the internal data from the current instance into the new `VTableD5` instance.
     *
     * @param shape The shape of the new `VTableD5` instance. Must contain exactly 4 integer dimensions.
     * @return A new `VTableD5` instance with the specified shape, containing data copied from the current instance.
     * @throws IllegalArgumentException If the provided shape does not have exactly 4 dimensions.
     */
    fun asVTable5(vararg shape: Int): VTableD5 =
        VTableD5(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current VTable instance into a new VTableDN instance with the specified shape.
     * Copies the internal data from the current instance into the newly created VTableDN instance.
     *
     * @param shape The shape of the new VTableDN instance. Must contain at least 2 dimensions.
     * @return A new VTableDN instance with the specified shape, containing data copied from the current instance.
     * @throws IllegalArgumentException If the provided shape contains fewer than 2 dimensions.
     */
    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}