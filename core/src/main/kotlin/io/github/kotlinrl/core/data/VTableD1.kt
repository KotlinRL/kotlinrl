package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.policy.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Represents a one-dimensional value table (VTable) implementation of the `EnumerableValueFunction` interface.
 * This class allows for mapping single-dimensional states to specific values and supports various operations,
 * including updating values, retrieving all states, computing the maximum value, and converting into higher-dimensional
 * VTable representations.
 *
 * @constructor Constructs a one-dimensional VTable with the specified shape.
 * The shape must have exactly one dimension.
 *
 * @property shape The shape of the VTable, defining its dimensions.
 *
 * @throws IllegalArgumentException when the provided shape does not have exactly one dimension.
 */
class VTableD1(
    vararg val shape: Int
) : EnumerableValueFunction<Int> {

    init {
        require(shape.size == 1) { "VTableD1 shape requires exactly 1 argument" }
    }

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
     * Updates the value associated with a specific state in the value table and returns the updated enumerable value function.
     *
     * @param state The state whose associated value is to be updated.
     * @param value The new value to associate with the specified state.
     * @return The updated instance of the enumerable value function.
     */
    override fun update(state: Int, value: Double): EnumerableValueFunction<Int> =
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
        VTableD1(*shape).also {
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
     * Converts the current value table into a two-dimensional value table representation (VTableD2)
     * with the specified shape. The data from the original table is copied into the new table.
     *
     * @param shape The dimensions for the new two-dimensional value table. Must contain exactly two integers.
     * @return A new instance of VTableD2 representing the value table with the specified shape.
     */
    fun asVTable2(vararg shape: Int): VTableD2 =
        VTableD2(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current value table into a three-dimensional value table representation (`VTableD3`)
     * with the specified shape. The data from the original table is copied into the new table.
     *
     * @param shape The dimensions for the new three-dimensional value table. Must consist of exactly three integers.
     * @return A new instance of `VTableD3` with the specified shape, containing copied data from the current table.
     */
    fun asVTable3(vararg shape: Int): VTableD3 =
        VTableD3(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current value table into a four-dimensional value table representation (`VTableD4`)
     * with the specified shape. The data from the original table is copied into the new table.
     *
     * @param shape The dimensions for the new four-dimensional value table. Must consist of exactly four integers.
     * @return A new instance of `VTableD4` with the specified shape, containing copied data from the current table.
     */
    fun asVTable4(vararg shape: Int): VTableD4 =
        VTableD4(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current value table into a five-dimensional value table representation (`VTableD5`)
     * with the specified shape. The data from the original table is copied into the new table.
     *
     * @param shape The dimensions for the new five-dimensional value table. Must consist of exactly four integers.
     * @return A new instance of `VTableD5` with the specified shape, containing copied data from the current table.
     */
    fun asVTable5(vararg shape: Int): VTableD5 =
        VTableD5(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current value table into an n-dimensional value table (`VTableDN`) with the specified shape.
     * The data from the original table is copied into the new n-dimensional table.
     *
     * @param shape The dimensions for the new n-dimensional value table. Must contain at least two integers.
     * @return A new instance of `VTableDN` with the specified shape, containing copied data from the current table.
     */
    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}