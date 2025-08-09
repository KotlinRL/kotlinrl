package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * A class representing a 4-dimensional value table. This class organizes data in a multidimensional array
 * with dimensions defined by rows, columns, layers, and features. It extends the `EnumerableValueFunction`
 * interface for multidimensional states represented as `NDArray<Int, D3>`.
 *
 * @constructor Initializes the 4-dimensional value table with the specified dimensions for rows, columns, layers, and features.
 * @param rowSize The number of rows in the value table.
 * @param colSize The number of columns in the value table.
 * @param layerSize The number of layers in the value table.
 * @param featureSize The number of features in the value table.
 */
class VTableD4(
    rowSize: Int,
    colSize: Int,
    layerSize: Int,
    featureSize: Int
) : EnumerableValueFunction<NDArray<Int, D3>> {

    /**
     * Defines the shape of the 4-dimensional table in the form of an integer array.
     * The dimensions are represented as `[rowSize, colSize, layerSize, featureSize]`.
     *
     * - `rowSize`: Represents the number of rows in the table.
     * - `colSize`: Represents the number of columns in the table.
     * - `layerSize`: Represents the number of layers in the table.
     * - `featureSize`: Represents the number of features in the table.
     */
    val shape = intArrayOf(rowSize, colSize, layerSize, featureSize)

    /**
     * Represents the base value for the `VTableD4` instance.
     * This is an internal property initialized as a `VTableDN` with the same shape as the `VTableD4` object.
     * It serves as a foundational data structure within the multidimensional value table system.
     */
    internal val base = VTableDN(shape = shape)

    /**
     * Retrieves the value associated with a given 4-dimensional state represented as `NDArray<Int, D3>`.
     *
     * @param state The 4-dimensional state to retrieve the value for. Must be an instance of `NDArray<Int, D3>`.
     * @return The value associated with the provided state as a `Double`.
     */
    override operator fun get(state: NDArray<Int, D3>): Double =
        base[state.asDNArray()]

    /**
     * Retrieves the value at the specified 4-dimensional position.
     *
     * @param row The row index of the desired position.
     * @param col The column index of the desired position.
     * @param layer The layer index of the desired position.
     * @param feature The feature index of the desired position.
     * @return The value at the specified position as a Double.
     */
    operator fun get(row: Int, col: Int, layer: Int, feature: Int): Double =
        this[mk.ndarray(mk[mk[mk[row, col, layer, feature]]])]


    /**
     * Updates the value associated with the specified 4-dimensional state in this `VTableD4` instance.
     * The state is identified by an `NDArray` of integers, and the provided value is applied at the mapped position.
     * A new instance of `VTableD4` with the updated value is returned.
     *
     * @param state The 4-dimensional state to update, represented as an `NDArray<Int, D3>` instance.
     * @param value The new value to assign to the specified state.
     * @return A new `VTableD4` instance containing the updated value at the specified state.
     */
    override fun update(state: NDArray<Int, D3>, value: Double): VTableD4 =
        copy().also { it.base.table[state.toIntArray()] = value }

    /**
     * Updates the value at the specified 4-dimensional position (row, column, layer, feature)
     * with the given value. The result is a new instance of `VTableD4` with the updated value.
     *
     * @param row The row index of the position to update.
     * @param col The column index of the position to update.
     * @param layer The layer index of the position to update.
     * @param feature The feature index of the position to update.
     * @param value The new value to assign to the specified position.
     * @return A new `VTableD4` instance with the updated value.
     */
    fun update(row: Int, col: Int, layer: Int, feature: Int, value: Double): VTableD4 =
        update(mk.ndarray(mk[mk[mk[row, col, layer, feature]]]), value)

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
        VTableD4(
            rowSize = shape[0],
            colSize = shape[1],
            layerSize = shape[2],
            featureSize = shape[3]
        ).also {
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
     * Converts the current 4-dimensional `VTableD4` instance into a 5-dimensional `VTableD5` instance.
     * The method creates a new `VTableD5` object with the specified dimensions, copies the data
     * from the current table into the new table, and returns the created instance.
     *
     * @param rowSize The size of the first dimension (rows) of the new `VTableD5` instance.
     * @param colSize The size of the second dimension (columns) of the new `VTableD5` instance.
     * @param layerSize The size of the third dimension (layers) of the new `VTableD5` instance.
     * @param featureSize The size of the fourth dimension (features) of the new `VTableD5` instance.
     * @param channelSize The size of the fifth dimension (channels) of the new `VTableD5` instance.
     * @return A `VTableD5` instance with the specified dimensions and the data copied from the current `VTableD4` instance.
     */
    fun asVTable5(rowSize: Int,
                  colSize: Int,
                  layerSize: Int,
                  featureSize: Int,
                  channelSize: Int): VTableD5 =
        VTableD5(
            rowSize = rowSize,
            colSize = colSize,
            layerSize = layerSize,
            featureSize = featureSize,
            channelSize = channelSize
            ).also {
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