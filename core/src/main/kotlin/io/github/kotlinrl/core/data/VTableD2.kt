package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * A two-dimensional value table (VTable) implementation that defines a mapping of states to values.
 * The class provides utilities for accessing, updating, and transforming the table structure while
 * maintaining compatibility with various dimensional formats.
 *
 * @constructor Creates a two-dimensional VTable instance with a predefined row and column size.
 * @param rowSize Number of rows in the table.
 * @param colSize Number of columns in the table.
 */
class VTableD2(
    rowSize: Int,
    colSize: Int,
) : EnumerableValueFunction<NDArray<Int, D1>> {

    /**
     * Represents the shape of the underlying data structure in the VTable, defined as a two-dimensional array.
     *
     * The `shape` variable is an integer array containing two elements:
     * - The number of rows in the table.
     * - The number of columns in the table.
     *
     * This variable is used to define the dimensions of the VTable for operations such as data access,
     * updates, and conversions to other VTable formats.
     */
    val shape = intArrayOf(rowSize, colSize)

    /**
     * Represents the base data structure of the VTableD2 instance.
     *
     * This internal value is an instance of `VTableDN` initialized with the shape specified by
     * the `shape` property of the containing `VTableD2` class. It serves as the underlying
     * data table that manages and manipulates state and value mappings.
     */
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
     * Retrieves the value at the specified row and column in the VTable.
     *
     * @param row The row index to access.
     * @param col The column index to access.
     * @return The `Double` value stored at the specified row and column.
     */
    operator fun get(row: Int, col: Int): Double =
        this[mk.ndarray(mk[row, col])]

    /**
     * Updates the VTable by setting the specified value at the given state.
     *
     * @param state The state represented as an NDArray of integers with one dimension (D1).
     * @param value The new value to set for the given state.
     * @return A new instance of VTableD2 with the updated state.
     */
    override fun update(state: NDArray<Int, D1>, value: Double): VTableD2 =
        copy().also { it.base.table[state.toIntArray()] = value }

    /**
     * Updates the VTable by setting a specific value at the given row and column.
     *
     * @param row The row index to update.
     * @param col The column index to update.
     * @param value The new value to set at the specified row and column.
     * @return A new instance of VTableD2 with the updated state.
     */
    fun update(row: Int, col: Int, value: Double): VTableD2 =
        update(mk.ndarray(mk[row, col]), value)

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
        VTableD2(rowSize = shape[0], colSize = shape[1]).also {
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
     * Converts the current `VTableD2` instance into a `VTableD3` with the specified dimensions.
     * Copies the internal data from the current instance into the new `VTableD3` instance.
     *
     * @param rowSize The number of rows in the resulting `VTableD3`.
     * @param colSize The number of columns in the resulting `VTableD3`.
     * @param layerSize The number of layers in the resulting `VTableD3`.
     * @return A new `VTableD3` instance with the specified dimensions, containing data copied from the current instance.
     */
    fun asVTable3(
        rowSize: Int,
        colSize: Int,
        layerSize: Int
    ): VTableD3 =
        VTableD3(
            rowSize = rowSize,
            colSize = colSize,
            layerSize = layerSize
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current instance into a `VTableD4` with the specified dimensions.
     * Copies the internal data from the current instance into the new `VTableD4`.
     *
     * @param rowSize The size of the rows in the resulting `VTableD4`.
     * @param colSize The size of the columns in the resulting `VTableD4`.
     * @param layerSize The number of layers in the resulting `VTableD4`.
     * @param featureSize The number of features in the resulting `VTableD4`.
     * @return A new `VTableD4` instance with the specified dimensions, containing data copied from the current instance.
     */
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
     * Converts the current instance into a new `VTableD5` with the specified dimensions.
     * Copies the internal data from the current instance into the new `VTableD5` instance.
     *
     * @param rowSize The size of the rows in the resulting `VTableD5`.
     * @param colSize The size of the columns in the resulting `VTableD5`.
     * @param layerSize The number of layers in the resulting `VTableD5`.
     * @param featureSize The number of features in the resulting `VTableD5`.
     * @param channelSize The number of channels in the resulting `VTableD5`.
     * @return A new `VTableD5` instance with the specified dimensions, containing data copied from the current instance.
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