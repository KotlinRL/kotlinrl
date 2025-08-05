package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * Represents a specialized 3-dimensional value table that conforms to the `EnumerableValueFunction` interface.
 * A `VTableD3` maintains an enumerable mapping of 3-dimensional states to their associated values.
 * Each state is represented by a 2-dimensional numerical array, and values are stored in a backing `VTableDN`.
 *
 * @constructor Initializes the `VTableD3` with a specific shape.
 * The shape must have exactly three dimensions. Throws an exception if the size of `shape` is not 3.
 *
 * @property shape The dimensions of the 3-dimensional value table.
 */
class VTableD3(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D2>> {

    init {
        require(shape.size == 3) { "VTableD3 shape requires exactly 3 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    /**
     * Retrieves the value associated with a specific 2-dimensional state from the current value table.
     *
     * @param state The 2-dimensional numerical array (NDArray) representing the state
     *              for which the value is to be retrieved.
     * @return The value of type Double corresponding to the provided state.
     */
    override fun get(state: NDArray<Int, D2>): Double =
        base[state.asDNArray()]

    /**
     * Updates the value in the value table for a specific 2-dimensional state.
     *
     * @param state The 2-dimensional numerical array (NDArray) representing the state
     *              for which the value needs to be updated.
     * @param value The new double value to be associated with the provided state.
     * @return An updated instance of EnumerableValueFunction containing the updated value table.
     */
    override fun update(state: NDArray<Int, D2>, value: Double): EnumerableValueFunction<NDArray<Int, D2>> =
        copy().also { it.base.table[state.toIntArray()] = value }

    /**
     * Retrieves all possible 2-dimensional states represented as NDArrays of integers.
     *
     * This method maps the states of higher dimensions from the base object to a 2-dimensional representation,
     * ensuring that the states conform to the dimensionality of D2 NDArray.
     *
     * @return A list of 2-dimensional states as NDArray<Int, D2>.
     */
    override fun allStates(): List<NDArray<Int, D2>> =
        base.allStates().map { it.asD2Array() }

    /**
     * Retrieves the maximum value from the current value table.
     *
     * @return The maximum value of type Double present in the value table.
     */
    override fun max(): Double =
        base.max()

    /**
     * Creates and returns a deep copy of the current instance of the value table.
     * The copy will have identical structure and data but will be independent of the original instance.
     *
     * @return A new instance of VTableD3 containing the same data as the current instance.
     */
    fun copy(): VTableD3 =
        VTableD3(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the value table to a CSV file at the specified path.
     *
     * This method utilizes a utility function to serialize the internal NDArray
     * representing the value table into a CSV format. The resulting file can
     * be accessed at the provided path location.
     *
     * @param path The file path where the value table will be saved in CSV format.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads the value table from a CSV file located at the specified path.
     *
     * This method reads the contents of the CSV file, converts it into an NDArray,
     * and reshapes it to match the internal shape of the value table. The reshaped
     * data is then copied into the current instance of the value table.
     *
     * @param path The file path to the CSV file which contains the value table data.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the internal value table of the VTableD3 instance.
     *
     * This method delegates the printing functionality to the `print` function of the `base` instance,
     * which represents the foundational data structure holding the value table.
     *
     * The printed output typically represents the current state of the table, formatted as a multidimensional array.
     */
    fun print() = base.print()

    /**
     * Converts the current value table to a 4-dimensional value table (`VTableD4`) with the specified shape.
     *
     * Creates a new instance of `VTableD4` with the provided shape arguments. The internal data is copied
     * from the current value table to the newly created value table.
     *
     * @param shape The dimensions for the 4-dimensional value table. Must contain exactly 4 arguments.
     * @return A new instance of `VTableD4` representing the converted 4-dimensional value table.
     */
    fun asVTable4(vararg shape: Int): VTableD4 =
        VTableD4(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current value table to a 5-dimensional value table (`VTableD5`) with the specified shape.
     *
     * This function creates a new instance of `VTableD5` with the provided shape arguments. The internal
     * data from the current value table is copied into the new 5-dimensional value table instance.
     *
     * @param shape The dimensions for the 5-dimensional value table. Must contain exactly 4 arguments.
     * @return A new instance of `VTableD5` representing the converted 5-dimensional value table.
     */
    fun asVTable5(vararg shape: Int): VTableD5 =
        VTableD5(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current value table into an N-dimensional value table (`VTableDN`) with the specified shape.
     *
     * This method creates a new instance of `VTableDN` with the provided shape dimensions. The internal data
     * from the current value table is copied into the newly created instance.
     *
     * @param shape The dimensions for the resulting N-dimensional value table. Must contain at least two elements.
     * @return A new instance of `VTableDN` representing the converted N-dimensional value table.
     */
    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}