package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

/**
 * Represents a Q-table implementation for deterministic or non-deterministic environments
 * where states and actions are described as an NDArray of integers. The table organizes
 * Q-values as an NDArray with a shape derived from the input dimensions, where the last
 * dimension corresponds to actions.
 *
 * @constructor Initializes a Q-table with a given shape, a deterministic or probabilistic
 * selection strategy for actions, a tolerance for approximate matches, and an optional
 * default Q-value.
 *
 * @param shape The dimensions of the Q-table, where the last dimension corresponds to the
 * available actions. Requires at least 2 dimensions (state dimensions + actions).
 * @param deterministic If `true`, selects actions deterministically based on the highest
 * Q-value. If `false`, considers approximate matches based on the tolerance.
 * @param tolerance If `deterministic` is `false`, allows actions within this range of the
 * highest Q-value to be considered equivalently.
 * @param defaultQValue The default Q-value assigned to all state-action pairs at initialization.
 *
 * @throws IllegalArgumentException if the shape has fewer than 2 dimensions.
 */
class QTableDN(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, DN>, Int> {

    init {
        require(shape.size >= 2) { "QTableDN shape requires at least 2 arguments" }
    }

    internal var table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { defaultQValue }.asDNArray()

    /**
     * Converts the current Q-table into a value function representation.
     *
     * The method iterates over all states, computes the maximum Q-value for each state,
     * and stores these as values in a value function.
     *
     * @return An instance of `EnumerableValueFunction` containing the maximum Q-values for all states.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): EnumerableValueFunction<NDArray<Int, DN>> {
        val Q = (if (deterministic) this else copy(true))
        val shape = Q.shape.dropLast(1).toIntArray()
        var V = VTableDN(shape = shape)
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state)) as VTableDN
        }
        return V
    }

    /**
     * Retrieves the Q-value for the given state and action combination from the Q-table.
     *
     * @param state The state represented as an NDArray of integers.
     * @param action The action represented as an integer.
     * @return The Q-value associated with the specified state and action.
     */
    override operator fun get(state: NDArray<Int, DN>, action: Int): Double =
        table[state.toIntArray() + action]

    /**
     * Updates the Q-value for a specific state-action pair in the Q-table.
     *
     * @param state The state represented as an NDArray of integers.
     * @param action The action represented as an integer.
     * @param value The Q-value to be updated for the given state and action.
     * @return A new instance of the QTableDN with the updated Q-value for the specified state-action pair.
     */
    override fun update(
        state: NDArray<Int, DN>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, DN>, Int> =
        copy().also { it.table[state.toIntArray() + action] = value }

    /**
     * Retrieves a list of all possible states within the multidimensional structure defined
     * by the Q-table's dimensions, excluding the action dimension.
     *
     * @return A list of states, where each state is represented as an NDArray of integers.
     */
    override fun allStates(): List<NDArray<Int, DN>> {
        val stateShape = shape.dropLast(1) // all but action dimension
        val rawStates = cartesianProduct(*stateShape.map { 0 until it }.toTypedArray())
        return rawStates.map { mk.ndarray(it).asDNArray() }
    }

    /**
     * Computes the Q-values for a given state from the Q-table.
     *
     * @param state The state represented as an NDArray of integers.
     * @return An NDArray of doubles containing the Q-values for the given state.
     */
    private fun qValues(state: NDArray<Int, DN>): NDArray<Double, D1> {
        val axes = IntArray(state.shape[0]) { it }
        return table.view(state.toIntArray(), axes).asDNArray().asD1Array()
    }

    /**
     * Finds the maximum Q-value for a given state in the Q-table.
     *
     * @param state The state represented as an NDArray of integers.
     * @return The maximum Q-value for the specified state as a double.
     */
    override fun maxValue(state: NDArray<Int, DN>): Double = qValues(state).max() ?: 0.0

    /**
     * Determines the best action to take for a given state based on the Q-values.
     * If the `deterministic` flag is true, it returns the action with the highest Q-value.
     * Otherwise, it selects an action probabilistically, favoring actions with similar Q-values
     * to the maximum within a specified tolerance.
     *
     * @param state The state represented as an NDArray of integers.
     * @return The selected action as an integer.
     */
    override fun bestAction(state: NDArray<Int, DN>): Int {
        val Q = qValues(state)
        return if (deterministic) {
            Q.argMax()
        } else {
            val max = Q.max() ?: 0.0
            val candidates = Q.indices.filter { abs(Q[it] - max) < tolerance }
            when {
                candidates.isNotEmpty() -> if (candidates.size > 1) candidates.random() else candidates.first()
                else -> Q.indices.random()
            }
        }
    }

    /**
     * Saves the current Q-table to a specified file path in CSV format.
     *
     * @param path The file path where the Q-table should be saved.
     */
    fun save(path: String) {
        mk.writeCsvSafely(path, table)
    }

    /**
     * Loads data from a CSV file into the Q-table and reshapes it according to the specified dimensions.
     *
     * @param path The file path from which the data is loaded. The file is expected to contain CSV data
     * formatted as needed for the Q-table.
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
     * Prints the Q-table to the standard output.
     *
     * This method outputs the contents of the internal Q-table, represented by
     * the `table` field, to the console. The `table` is expected to hold the
     * multidimensional array of Q-values used for storing state-action relationships.
     */
    fun print() = println(table)

    /**
     * Creates a copy of the current QTableDN instance with an optional override for the deterministic parameter.
     *
     * @param deterministic A boolean indicating whether the copied instance should use deterministic behavior.
     *                       If not provided, it defaults to the value of `this.deterministic`.
     * @return A new QTableDN instance with the specified `deterministic` value and the same attributes
     *         and data as the current instance.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableDN =
        QTableDN(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            table.data.copyInto(it.table.data)
        }

    /**
     * Computes the Cartesian product of multiple input ranges.
     *
     * This method takes a variable number of iterable ranges, where each range is an iterable collection of integers,
     * and generates all possible combinations of integers, one from each range, arranged as arrays.
     *
     * @param ranges A variable number of iterable ranges, each representing a collection of integers.
     * @return A list of integer arrays, where each array represents one combination of integers selected from the input ranges.
     */
    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}