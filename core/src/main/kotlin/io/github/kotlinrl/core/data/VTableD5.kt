package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * Represents a specialized value table for a 4-dimensional state space with an additional shape-level abstraction.
 * This class is designed to store and manage value functions for states represented as `NDArray<Int, D4>`.
 * It provides operations to access, update, and manipulate the values associated with each state in an enumerable state space.
 *
 * This table is a specific extension of `VTableDN`, which supports arbitrary-dimensional state spaces.
 * The implementation enforces a shape of size 4 for the state space and maps into a general-purpose `VTableDN` for internal operations.
 *
 * @constructor Initializes a new instance of the `VTableD5` class with the specified shape.
 * The provided shape must be an array of 4 integers.
 * @param shape The dimensions of the `VTableD5`. Must contain exactly 4 integers.
 *
 * @throws IllegalArgumentException If the provided shape does not have exactly 4 integers.
 */
class VTableD5(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D4>> {

    init {
        require(shape.size == 4) { "VTableD5 shape requires exactly 4 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    /**
     * Retrieves the value associated with the given state in the underlying data structure.
     *
     * @param state The NDArray representing the state for which the value is to be retrieved.
     * @return The value corresponding to the provided state as a Double.
     */
    override fun get(state: NDArray<Int, D4>): Double =
        base[state.asDNArray()]

    /**
     * Updates the value associated with the given state in the underlying data structure
     * and returns an updated instance of `EnumerableValueFunction`.
     *
     * @param state The NDArray representing the state for which the value is to be updated.
     * @param value The new value to associate with the given state.
     * @return An updated instance of `EnumerableValueFunction<NDArray<Int, D4>>` reflecting the changes.
     */
    override fun update(state: NDArray<Int, D4>, value: Double): EnumerableValueFunction<NDArray<Int, D4>> =
        copy().also { it.base.table[state.toIntArray()] = value }

    /**
     * Retrieves all possible states represented as a list of 4-dimensional NDArrays.
     *
     * This method converts the states obtained from the base data structure into the
     * specific type `NDArray<Int, D4>`, ensuring compatibility with 4-dimensional data.
     *
     * @return A list of `NDArray<Int, D4>` containing all possible states.
     */
    override fun allStates(): List<NDArray<Int, D4>> =
        base.allStates().map { it.asD4Array() }

    /**
     * Computes and retrieves the maximum value from the underlying data structure.
     *
     * @return The maximum value as a Double.
     */
    override fun max(): Double =
        base.max()

    /**
     * Creates and returns a copy of the current `VTableD5` instance.
     *
     * The method constructs a new `VTableD5` object with the same shape as the current instance and
     * copies the underlying data from the current instance to the new one.
     *
     * @return A new `VTableD5` instance that is a copy of the current object.
     */
    fun copy(): VTableD5 =
        VTableD5(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the underlying table data to a file at the specified path in CSV format.
     *
     * @param path The file path where the data will be saved.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads table data from a CSV file and reshapes it to match the dimensions of the current table.
     * The loaded data is copied into the underlying data structure.
     *
     * @param path The file path to the CSV file containing the table data.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the underlying table data to the standard output.
     *
     * This method leverages the `print()` function of the base data structure
     * to output the current state of the table in a human-readable format.
     */
    fun print() = base.print()

    /**
     * Converts the current `VTableD5` instance into a `VTableDN` instance with the specified shape.
     * This method creates a new `VTableDN` object, transfers the underlying data from
     * the base table of the current instance into the new instance, and returns it.
     *
     * @param shape The desired shape for the new `VTableDN` instance. The shape must contain at least two dimensions.
     * @return A new `VTableDN` instance with the specified shape and data copied from the base table of the current instance.
     */
    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}