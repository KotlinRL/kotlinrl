package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * A one-dimensional implementation of `EnumerableQFunction`, designed to efficiently represent
 * and manage Q-values for a system with two state-action dimensionality. This class simplifies
 * the Q-function table by assuming the state space is one-dimensional.
 *
 * The underlying Q-table is internally represented through a `QTableDN` instance, enabling
 * additional flexibility and extensibility for dimensional transformations.
 *
 * @property shape Specifies the dimensions of the state and action space. Requires exactly two integers.
 * @property deterministic Determines whether the action selection process is deterministic or probabilistic.
 * @property tolerance The allowed numerical tolerance for comparing Q-values in non-deterministic selection.
 * @property defaultQValue The initial value assigned to all Q-values in the table.
 */
class QTableD1(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<Int, Int> {

    init {
        require(shape.size == 2) { "QTableD1 shape requires exactly 2 arguments" }
    }

    private val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Retrieves the Q-value for the given state and action from the QTable.
     *
     * @param state The integer representation of the state for which the Q-value is to be retrieved.
     * @param action The integer index of the action for which the Q-value is to be retrieved.
     * @return The Q-value corresponding to the specified state and action.
     */
    override fun get(state: Int, action: Int): Double =
        base[mk.ndarray(intArrayOf(state)).asDNArray(), action]

    /**
     * Updates the Q-value for the specified state and action in the QTable.
     *
     * @param state The integer representation of the state for which the Q-value is to be updated.
     * @param action The integer index of the action for which the Q-value is to be updated.
     * @param value The new Q-value to set for the specified state and action.
     * @return A new instance of the QTableD1 with the updated Q-value for the specified state and action.
     */
    override fun update(
        state: Int,
        action: Int,
        value: Double
    ): EnumerableQFunction<Int, Int> =
        copy().also { it.base.table[intArrayOf(state) + action] = value }

    /**
     * Retrieves a list of all possible states, represented as integers, in the Q-table.
     *
     * @return A list of integers, where each integer represents a possible state in the Q-table.
     */
    override fun allStates(): List<Int> =
        base.allStates().map { it[0] }

    /**
     * Calculates the maximum Q-value for a given state.
     *
     * @param state The integer representation of the state for which the maximum Q-value is to be retrieved.
     * @return The maximum Q-value corresponding to the specified state.
     */
    override fun maxValue(state: Int): Double =
        base.maxValue(mk.ndarray(intArrayOf(state)).asDNArray())

    /**
     * Determines the best action to take for a given state based on the Q-values.
     *
     * @param state The integer representation of the state for which the best action is to be determined.
     * @return The index of the best action to take for the specified state.
     */
    override fun bestAction(state: Int): Int =
        base.bestAction(mk.ndarray(intArrayOf(state)).asDNArray())

    /**
     * Creates a duplicate instance of the current QTableD1 object, maintaining the same configuration and state.
     *
     * @return A new QTableD1 instance with the same attributes and internal data as the original object.
     */
    fun copy(): QTableD1 =
        QTableD1(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the current Q-table data to the specified file path in CSV format.
     *
     * @param path The file path where the Q-table will be saved.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads a Q-table's data from a CSV file specified by the provided path. The data is reshaped
     * according to the dimensions of the Q-table.
     *
     * @param path The file path to the CSV file containing the Q-table data.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the contents of the underlying Q-table.
     * Delegates the print operation to the base Q-table implementation.
     */
    fun print() = base.print()

    /**
     * Converts the current QTableD1 instance into a QTableD2 instance with the specified shape.
     *
     * @param shape A variable number of integers specifying the shape of the resulting QTableD2.
     *              It must contain exactly three dimensions.
     * @return A new QTableD2 instance with the specified shape, initialized with the current Q-table's attributes and data.
     */
    fun asQTableD2(vararg shape: Int): QTableD2 =
        QTableD2(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current `QTableD1` instance into a `QTableD3` instance with the specified shape.
     *
     * @param shape A variable number of integers specifying the shape of the resulting `QTableD3`.
     *              It must contain exactly 4 dimensions.
     * @return A new `QTableD3` instance with the specified shape, initialized with the current Q-table's attributes and data.
     */
    fun asQTableD3(vararg shape: Int): QTableD3 =
        QTableD3(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current `QTableD1` instance into a `QTableD4` instance with the specified shape.
     *
     * @param shape A variable number of integers specifying the shape of the resulting `QTableD4`.
     *              It must contain exactly 5 dimensions.
     * @return A new `QTableD4` instance with the specified shape, initialized with the current Q-table's attributes and data.
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
     * Converts the current QTableD1 instance into a QTableD5 instance with the specified shape.
     *
     * @param shape A variable number of integers specifying the shape of the resulting QTableD5.
     *              It must contain exactly six dimensions.
     * @return A new QTableD5 instance with the specified shape, initialized with the current Q-table's attributes and data.
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
     * Converts the current `QTableD1` instance into a `QTableDN` instance with the specified shape.
     *
     * @param shape A variable number of integers specifying the shape of the resulting `QTableDN`.
     *              The shape must contain at least two dimensions.
     * @return A new `QTableDN` instance with the specified shape, initialized with the current Q-table's attributes and data.
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
