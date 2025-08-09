package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * A multi-dimensional table representation with 5 dimensions. Each dimension is specified
 * by its size upon initialization. The class provides methods to retrieve, update, and manipulate
 * values based on the table's 5-dimensional structure.
 *
 * This implementation stores data in an underlying `VTableDN` instance and supports operations
 * such as querying maximum values, retrieving all states, saving and loading table data, and
 * converting the table into a more generalized representation.
 *
 * @param rowSize The size of the row dimension of the table.
 * @param colSize The size of the column dimension of the table.
 * @param layerSize The size of the layer dimension of the table.
 * @param featureSize The size of the feature dimension of the table.
 * @param channelSize The size of the channel dimension of the table.
 */
class VTableD5(
    rowSize: Int,
    colSize: Int,
    layerSize: Int,
    featureSize: Int,
    channelSize: Int
) : EnumerableValueFunction<NDArray<Int, D4>> {

    /**
     * Represents the shape of a 5-dimensional data structure in the form of an integer array.
     *
     * Each element in the array corresponds to the size of a specific dimension:
     * - The first element (`rowSize`) defines the number of rows.
     * - The second element (`colSize`) defines the number of columns.
     * - The third element (`layerSize`) defines the number of layers.
     * - The fourth element (`featureSize`) defines the number of features.
     * - The fifth element (`channelSize`) defines the number of channels.
     */
    val shape = intArrayOf(rowSize, colSize, layerSize, featureSize, channelSize)

    /**
     * Represents the underlying base table data structure used by the `VTableD5` class.
     * This internal value is initialized as a `VTableDN` instance with a shape corresponding
     * to the dimensions of the current multi-dimensional table.
     *
     * This field provides the storage mechanism for the table and supports operations such as retrieving,
     * updating, and transforming table data.
     */
    internal val base = VTableDN(shape = shape)

    /**
     * Retrieves the value associated with the given state in the underlying data structure.
     *
     * @param state The NDArray representing the state for which the value is to be retrieved.
     * @return The value corresponding to the provided state as a Double.
     */
    override operator fun get(state: NDArray<Int, D4>): Double =
        base[state.asDNArray()]

    /**
     * Retrieves the value at the specified multi-dimensional coordinates in the table.
     *
     * @param row The index of the row dimension.
     * @param col The index of the column dimension.
     * @param layer The index of the layer dimension.
     * @param feature The index of the feature dimension.
     * @param channel The index of the channel dimension.
     * @return The value located at the specified coordinates as a Double.
     */
    operator fun get(row: Int, col: Int, layer: Int, feature: Int, channel: Int): Double =
        this[mk.ndarray(mk[mk[mk[mk[row, col, layer, feature, channel]]]])]

    /**
     * Updates the value at the specified state in the underlying data structure and returns
     * an updated instance of `VTableD5`.
     *
     * @param state The NDArray representing the state to be updated. The indices in the array
     * correspond to the multi-dimensional coordinates of the table.
     * @param value The new value to be set at the specified state.
     * @return An updated instance of `VTableD5` reflecting the change in value at the specified state.
     */
    override fun update(state: NDArray<Int, D4>, value: Double): VTableD5 =
        copy().also { it.base.table[state.toIntArray()] = value }

    /**
     * Updates the value at the specified multi-dimensional coordinates in the underlying data structure
     * and returns an updated instance of `VTableD5`.
     *
     * @param row The index of the row dimension.
     * @param col The index of the column dimension.
     * @param layer The index of the layer dimension.
     * @param feature The index of the feature dimension.
     * @param channel The index of the channel dimension.
     * @param value The new value to associate with the specified location.
     * @return An updated instance of `VTableD5` reflecting the changes at the specified location.
     */
    fun update(row: Int, col: Int, layer: Int, feature: Int, channel: Int, value: Double): VTableD5 =
        update(mk.ndarray(mk[mk[mk[mk[row, col, layer, feature, channel]]]]), value)

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
        VTableD5(
            rowSize = shape[0],
            colSize = shape[1],
            layerSize = shape[2],
            featureSize = shape[3],
            channelSize = shape[4]
        ).also {
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