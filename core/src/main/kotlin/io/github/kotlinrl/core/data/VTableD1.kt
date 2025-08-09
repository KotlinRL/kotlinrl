package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.policy.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * A specific implementation of `EnumerableValueFunction` for a one-dimensional value table.
 * This class manages the state-value mapping and provides operations for retrieving, updating, and
 * manipulating values in the context of a single-dimensional state space.
 *
 * @constructor Initializes the one-dimensional value table with a specified state size.
 * @param state The size of the state space for the one-dimensional value table.
 */
class VTableD1(
    state: Int
) : EnumerableValueFunction<Int> {

    /**
     * Represents the shape of the underlying value table as an integer array.
     * The shape defines the dimensional structure of the table, where each element in the array
     * corresponds to the size of a dimension in a multi-dimensional structure.
     */
    val shape = intArrayOf(state)

    /**
     * Represents the base n-dimensional value table for the `VTableD1` class.
     * It is initialized with the same shape as the parent instance.
     * This serves as the foundational data structure that the class methods operate upon.
     */
    private val base = VTableDN(shape = shape)

    /**
     * Retrieves the value associated with the specified state.
     *
     * @param state The integer value representing the state for which the value is retrieved.
     * @return The value associated with the given state as a Double.
     */
    override fun get(state: Int): Double =
        base[mk.ndarray(intArrayOf(state)).asDNArray()]

    /**
     * Updates the value associated with a specific state in the value table.
     *
     * @param state The integer value representing the state to be updated.
     * @param value The new value to assign to the specified state as a Double.
     * @return A new instance of `VTableD1` with the updated value for the specified state.
     */
    override fun update(state: Int, value: Double): VTableD1 =
        copy().also { it.base.table[intArrayOf(state)] = value }

    /**
     * Retrieves a list of integers representing the first element of each state
     * in the underlying multi-dimensional state representation.
     *
     * @return A list of integers where each integer corresponds to the first element
     *         of each state in the Cartesian product of the base dimensions.
     */
    override fun allStates(): List<Int> =
        base.allStates().map { it[0] }

    /**
     * Finds the maximum value in the underlying data.
     *
     * @return The maximum value as a Double.
     */
    override fun max(): Double =
        base.max()

    /**
     * Creates a new instance of `VTableD1` with the same shape as the current instance,
     * and copies the data from the base table of the current instance into the new instance.
     *
     * @return A new `VTableD1` object containing the copied data from the current instance.
     */
    fun copy(): VTableD1 =
        VTableD1(shape[0]).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the current state of the value table to a CSV file at the specified path.
     *
     * @param path The file path where the value table data will be saved.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads the value table data from a CSV file at the specified path and reshapes it
     * based on the shape of the current instance. The reshaped data is then copied
     * into the underlying table's data structure.
     *
     * @param path The file path from which the value table data will be loaded.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the underlying table or data structure associated with the current instance
     * to the standard output.
     */
    fun print() = base.print()

    /**
     * Converts the current one-dimensional value table into a two-dimensional value table (`VTableD2`)
     * with the specified row and column sizes. The data from the original table is copied into the new table.
     *
     * @param rowSize The number of rows in the new two-dimensional value table.
     * @param colSize The number of columns in the new two-dimensional value table.
     * @return A new instance of `VTableD2` with the specified dimensions, containing copied data from the current table.
     */
    fun asVTable2(rowSize: Int, colSize: Int): VTableD2 =
        VTableD2(rowSize = rowSize, colSize = colSize).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current one-dimensional value table into a three-dimensional value table (`VTableD3`),
     * with the specified row, column, and layer sizes. The data from the original table is copied into the new table.
     *
     * @param rowSize The number of rows in the new three-dimensional value table.
     * @param colSize The number of columns in the new three-dimensional value table.
     * @param layerSize The number of layers in the new three-dimensional value table.
     * @return A new instance of `VTableD3` with the specified dimensions, containing copied data from the current table.
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
     * Converts the current value table into a four-dimensional value table (`VTableD4`)
     * with the specified row, column, layer, and feature sizes. The data from the original
     * table is copied into the new table.
     *
     * @param rowSize The number of rows in the new four-dimensional value table.
     * @param colSize The number of columns in the new four-dimensional value table.
     * @param layerSize The number of layers in the new four-dimensional value table.
     * @param featureSize The number of features in the new four-dimensional value table.
     * @return A new instance of `VTableD4` with the specified dimensions, containing copied data from the current table.
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
     * Converts the current value table into a five-dimensional value table (`VTableD5`)
     * with the specified row, column, layer, feature, and channel sizes. The data from
     * the original table is copied into the new table.
     *
     * @param rowSize The number of rows in the new five-dimensional value table.
     * @param colSize The number of columns in the new five-dimensional value table.
     * @param layerSize The number of layers in the new five-dimensional value table.
     * @param featureSize The number of features in the new five-dimensional value table.
     * @param channelSize The number of channels in the new five-dimensional value table.
     * @return A new instance of `VTableD5` with the specified dimensions, containing copied data from the current table.
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
     * Converts the current value table into an N-dimensional value table (`VTableDN`) with the specified dimensions.
     * The data from the original value table is copied into the newly created table.
     *
     * @param shape A variable number of integers representing the dimensions of the new N-dimensional value table.
     * @return A new instance of `VTableDN` with the specified dimensions, containing copied data from the current value table.
     */
    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}