package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.policy.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * Represents a value table for discrete multi-dimensional states in a reinforcement learning context,
 * implemented as an `NDArray`. This class supports state-value lookups, updates, and serialization.
 *
 * @constructor Initializes a VTableDN with a specific shape, defining the dimensions of the state space.
 * @param shape A variable number of integers specifying the dimensions of the state space. At least two dimensions are required.
 *
 * @throws IllegalArgumentException If the provided shape has fewer than two dimensions.
 */
class VTableDN(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, DN>> {

    init {
        require(shape.size >= 2) { "VTableDN shape requires at least 2 arguments" }
    }

    internal val table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { 0.0 }.asDNArray()

    /**
     * Retrieves the value associated with the specified state from the VTableDN.
     *
     * @param state The multi-dimensional array (NDArray) of integers representing the state for which
     * the value is to be retrieved.
     * @return The value of type Double associated with the given state.
     */
    override operator fun get(state: NDArray<Int, DN>): Double = table[state.toIntArray()]

    /**
     * Updates the value associated with the specified state in the VTableDN and returns
     * an updated enumerable value function.
     *
     * @param state An NDArray of integers representing the state whose value is
     * to be updated. The state corresponds to a unique entry in the table.
     * @param value The new value to associate with the specified state in the table.
     * @return An updated instance of the enumerable value function reflecting
     * the new state-value mapping.
     */
    override fun update(
        state: NDArray<Int, DN>,
        value: Double
    ): EnumerableValueFunction<NDArray<Int, DN>> = copy().also { it.table[state.toIntArray()] = value }

    /**
     * Finds the maximum value from the data stored in the underlying table.
     *
     * @return The maximum value as a Double from the table's data.
     */
    override fun max(): Double = table.data.max()

    /**
     * Retrieves all possible states represented by the Cartesian product of the ranges
     * defined by the dimensions in the shape of the VTableDN.
     *
     * @return A list of NDArrays of integers, where each NDArray represents a unique
     *         state in the multi-dimensional table.
     */
    override fun allStates(): List<NDArray<Int, DN>> {
        val rawStates = cartesianProduct(*shape.map { 0 until it }.toTypedArray())
        return rawStates.map { mk.ndarray(it).asDNArray() }
    }

    /**
     * Creates and returns a copy of the current VTableDN instance.
     * The new instance has the same shape and contains identical data as the original instance.
     *
     * @return A new VTableDN instance with copied data and shape identical to the original.
     */
    fun copy(): VTableDN {
        return VTableDN(*shape).also { table.data.copyInto(it.table.data) }
    }

    /**
     * Saves the current state of the multi-dimensional table to a CSV file at the specified file path.
     *
     * @param path The file path where the table data will be saved as a CSV file.
     */
    fun save(path: String) {
        mk.writeCsvSafely(path, table)
    }

    /**
     * Loads data from a CSV file at the specified path and reshapes it into the dimensions
     * defined by the `shape` property of the current instance. The reshaped data is then
     * copied into the underlying table structure.
     *
     * @param path The file path to the CSV file to be loaded.
     */
    @Suppress("DuplicatedCode")
    fun load(path: String) {
        val dn = mk.readCsvSafely(path)
        val reshaped = when (shape.size) {
            2 -> dn.reshape(shape[0], shape[1])
            3 -> dn.reshape(shape[0], shape[1], shape[2])
            4 -> dn.reshape(shape[0], shape[1], shape[2], shape[3])
            else -> dn.reshape(shape[0], shape[1], shape[2], shape[3], *shape.copyOfRange(4, shape.size))
        }.asDNArray()
        reshaped.data.copyInto(table.data)
    }

    /**
     * Prints the contents of the underlying table of the VTableDN instance to the standard output.
     */
    fun print() = println(table)

    /**
     * Generates the Cartesian product of the provided ranges as a list of integer arrays.
     *
     * @param ranges A variable number of Iterable<Int> representing the ranges of values across different dimensions.
     * Each range will contribute to the Cartesian product.
     * @return A list of integer arrays, where each array represents one combination of the Cartesian product.
     */
    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}
