package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * Represents a three-dimensional implementation of a Q-table for reinforcement learning.
 * It provides functionality to interact with a 3D state-action space, update Q-values,
 * retrieve the best actions or maximum Q-values for given states, and convert to higher-dimensional tables.
 *
 * @constructor Initializes a 3D Q-table with the provided shape, deterministic policy flag,
 * tolerance for non-deterministic action selection, and default Q-value.
 * @param shape The dimensions of the state-action table. Must contain exactly 4 integers.
 * @param deterministic If true, selects the best action deterministically based on maximum Q-value.
 * @param tolerance A small value for distinguishing near-equal Q-values during action selection.
 * @param defaultQValue The initial Q-value assigned to all state-action pairs.
 */
class QTableD3(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D2>, Int> {

    init {
        require(shape.size == 4) { "QTableD3 shape requires exactly 4 arguments" }
    }

    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Converts the current QTableD3 instance into an EnumerableValueFunction representing the value function.
     * This method calculates the maximum Q-value for each state and updates the resulting value function accordingly.
     *
     * @return An EnumerableValueFunction instance that maps states to their respective maximum Q-values.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): EnumerableValueFunction<NDArray<Int, D2>> {
        val Q = (if (deterministic) this else copy(true))
        val shape = Q.shape.dropLast(1).toIntArray()
        var V = VTableD3(shape = shape)
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state)) as VTableD3
        }
        return V
    }

    /**
     * Retrieves the Q-value corresponding to the given state and action.
     *
     * @param state The state represented as a 2-dimensional NDArray of integers.
     * @param action The action represented as an integer.
     * @return The Q-value as a Double corresponding to the specified state and action.
     */
    override fun get(state: NDArray<Int, D2>, action: Int): Double = base[state.asDNArray(), action]

    /**
     * Updates the Q-value associated with a particular state and action pair.
     *
     * @param state The state represented as a 2-dimensional NDArray of integers.
     * @param action The action represented as an integer.
     * @param value The new Q-value to assign for the specified state-action pair.
     * @return A new instance of the EnumerableQFunction containing the updated Q-values.
     */
    override fun update(
        state: NDArray<Int, D2>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, D2>, Int> =
        copy().also { it.base.table[state.toIntArray() + action] = value }

    /**
     * Retrieves all possible states represented as 2-dimensional NDArrays of integers.
     *
     * @return A list of all states, where each state is represented as an NDArray with a 2-dimensional shape.
     */
    override fun allStates(): List<NDArray<Int, D2>> =
        base.allStates().map { it.asD2Array() }

    /**
     * Determines the maximum Q-value for a given state.
     *
     * @param state The state represented as a 2-dimensional NDArray of integers.
     * @return The maximum Q-value as a Double for the specified state.
     */
    override fun maxValue(state: NDArray<Int, D2>): Double =
        base.maxValue(state.asDNArray())

    /**
     * Determines the best action to take for a given state based on the Q-values.
     *
     * @param state The state represented as a 2-dimensional NDArray of integers.
     * @return The action, represented as an integer, that is determined to be the best for the specified state.
     */
    override fun bestAction(state: NDArray<Int, D2>): Int =
        base.bestAction(state.asDNArray())

    /**
     * Creates a copy of the current QTableD3 instance with an optionally updated deterministic property.
     *
     * @param deterministic An optional Boolean value to set the deterministic property of the copied instance.
     *                       Defaults to the deterministic property of the current instance.
     * @return A new QTableD3 instance with the same configuration and data as the original, except for
     *         the deterministic property if overridden.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableD3 =
        QTableD3(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the Q-table to a specified file path in CSV format.
     *
     * @param path The file path where the Q-table should be saved. The data will be exported safely using a CSV writer.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads the Q-table data from a specified CSV file into the current instance.
     *
     * @param path The file path of the CSV file to load. The CSV file should contain data
     *             compatible with the shape of the current Q-table.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the underlying table data of the QTableD3 instance.
     * This method delegates the printing operation to a base-level `print` function,
     * ensuring the table's contents are displayed in a user-readable format.
     */
    fun print() = base.print()

    /**
     * Converts the current QTableD3 instance to a QTableD4 instance with the specified shape.
     *
     * @param shape The shape of the new QTableD4 instance. Must contain exactly 5 integer values.
     * @return A new QTableD4 instance with the specified shape, preserving the properties and data from the current QTableD3 instance.
     */
    fun asQTableD4(vararg shape: Int): QTableD4 =
        QTableD4(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current QTableD3 instance to a QTableD5 instance with the specified shape.
     *
     * @param shape The shape of the new QTableD5 instance. Must contain exactly 6 integer values.
     * @return A new QTableD5 instance with the specified shape, preserving the properties and data from the current QTableD3 instance.
     */
    fun asQTableD5(vararg shape: Int): QTableD5 =
        QTableD5(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current QTableD3 instance into a QTableDN instance with the specified shape.
     *
     * @param shape The desired shape of the resulting QTableDN instance. This parameter accepts a variable number of integer values.
     * @return A new QTableDN instance with the specified shape, preserving the properties and data from the original QTableD3 instance.
     */
    fun asQTableDN(vararg shape: Int): QTableDN =
        QTableDN(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.table.data)
        }
}
