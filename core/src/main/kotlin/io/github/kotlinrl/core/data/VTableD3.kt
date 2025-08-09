package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * Represents a three-dimensional value table with operations and utilities for managing and transforming
 * data within a structured grid-like format.
 *
 * This class is designed to handle and manipulate three-dimensional data, with each dimension corresponding
 * to rows, columns, and layers. It provides methods for data retrieval, updates, state enumeration, and
 * dimensional transformations, serving use cases that require multi-dimensional numerical computation or analysis.
 *
 * The underlying data structure for the value table is based on an internal implementation of `VTableDN`,
 * and the class offers a range of operations optimized for three-dimensional contexts.
 *
 * @param rowSize The size of the first dimension representing the number of rows.
 * @param colSize The size of the second dimension representing the number of columns.
 * @param layerSize The size of the third dimension representing the number of layers.
 */
class VTableD3(
    rowSize: Int,
    colSize: Int,
    layerSize: Int
) : EnumerableValueFunction<NDArray<Int, D2>> {

    /**
     * Represents the shape of the underlying 3-dimensional value table.
     *
     * This array specifies the dimensions of the value table, where:
     * - The first element corresponds to the number of rows.
     * - The second element corresponds to the number of columns.
     * - The third element corresponds to the number of layers.
     */
    val shape = intArrayOf(rowSize, colSize, layerSize)

    /**
     * Represents the foundational data structure that supports the functionality of the VTableD3 class.
     *
     * Holds an instance of `VTableDN` initialized with the given shape. This serves as the underlying value table,
     * enabling higher-dimensional operations and transformations within the context of the `VTableD3` class.
     *
     * The `base` variable provides core data manipulation capabilities, such as retrieving, updating, and transforming
     * the value table across multiple dimensions. It is integral to the class's implementation and its ability to manage
     * complex multi-dimensional state representations and operations.
     */
    internal val base = VTableDN(shape = shape)

    /**
     * Retrieves the value associated with a specific 2-dimensional state from the current value table.
     *
     * @param state The 2-dimensional numerical array (NDArray) representing the state
     *              for which the value is to be retrieved.
     * @return The value of type Double corresponding to the provided state.
     */
    override operator fun get(state: NDArray<Int, D2>): Double =
        base[state.asDNArray()]

    /**
     * Retrieves the value from the value table at the specified row, column, and layer.
     *
     * @param row The row index of the value to be retrieved.
     * @param col The column index of the value to be retrieved.
     * @param layer The layer index of the value to be retrieved.
     * @return The value of type Double at the specified row, column, and layer indices.
     */
    operator fun get(row: Int, col: Int, layer: Int): Double =
        this[mk.ndarray(mk[mk[row, col, layer]])]

    /**
     * Updates the value in the value table for the specified 2-dimensional state.
     *
     * @param state The 2-dimensional numerical array (NDArray) representing the state
     *              whose value needs to be updated.
     * @param value The new value to assign to the specified state.
     * @return A new instance of VTableD3 with the updated value.
     */
    override fun update(state: NDArray<Int, D2>, value: Double): VTableD3 =
        copy().also { it.base.table[state.toIntArray()] = value }


    /**
     * Updates the value in the value table at the specified row, column, and layer with the given value.
     *
     * @param row The row index where the value needs to be updated.
     * @param col The column index where the value needs to be updated.
     * @param layer The layer index where the value needs to be updated.
     * @param value The new value to be set at the specified position.
     * @return A new instance of VTableD3 with the updated value.
     */
    fun update(row: Int, col: Int, layer: Int, value: Double): VTableD3 =
        update(mk.ndarray(mk[mk[row, col, layer]]), value)

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
        VTableD3(
            rowSize = shape[0],
            colSize = shape[1],
            layerSize = shape[2]
        ).also {
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
     * Converts a 3-dimensional value table into a 4-dimensional value table with the specified dimensions.
     *
     * This method creates a new instance of `VTableD4` with the provided dimensions (`rowSize`, `colSize`, `layerSize`, `featureSize`).
     * The internal data of the current value table is copied into the new instance to ensure consistency.
     *
     * @param rowSize The size of the first dimension (rows) in the resulting 4-dimensional value table.
     * @param colSize The size of the second dimension (columns) in the resulting 4-dimensional value table.
     * @param layerSize The``` sizek ofotlin the
    third*/
    fun asVTable4(
        rowSize: Int,
        colSize: Int,
        layerSize: Int,
        featureSize: Int
    ): VTableD4 =
        VTableD4(
            rowSize = rowSize,
            colSize = colSize,
            layerSize = layerSize,
            featureSize = featureSize
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts a 3-dimensional value table into a 5-dimensional value table with the specified dimensions.
     *
     * This method creates a new instance of `VTableD5` with the provided dimensions (`rowSize`, `colSize`,
     * `layerSize`, `featureSize`, `channelSize`). The internal data of the current value table is copied
     * into the new instance to ensure consistency.
     *
     * @param rowSize The size of the first dimension (rows) in the resulting 5-dimensional value table.
     * @param colSize The size of the second dimension (columns) in the resulting 5-dimensional value table.
     * @param layerSize The size of the third dimension (layers) in the resulting 5-dimensional value table.
     * @param featureSize The size of the fourth dimension (features) in the resulting 5-dimensional value table.
     * @param channelSize The size of the fifth dimension (channels) in the resulting 5-dimensional value table.
     * @return A new instance of `VTableD5` representing the converted 5-dimensional value table.
     */
    fun asVTable5(
        rowSize: Int,
        colSize: Int,
        layerSize: Int,
        featureSize: Int,
        channelSize: Int
    ): VTableD5 =
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