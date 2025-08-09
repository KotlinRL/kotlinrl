package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * A specialized implementation of the `EnumerableQFunction` interface representing a 2-dimensional
 * Q-table for reinforcement learning tasks. This class uses a structured representation of states
 * and actions, providing functionality for managing Q-values in a deterministic or stochastic manner.
 *
 * @property deterministic A flag indicating whether the Q-table should operate deterministically. If true,
 * the best action is selected deterministically; otherwise, stochastic behavior is applied for ties.
 * @property tolerance The allowable tolerance for determining equality of floating-point numbers, typically
 * used when comparing Q-values to resolve ties non-deterministically.
 * @property defaultQValue The initial Q-value assigned to all state-action pairs when the Q-table is instantiated.
 */
class QTableD2(
    rowSize: Int,
    colSize: Int,
    actionSize: Int,
    val deterministic: Boolean = true,
    val tolerance: Double = 1e-6,
    val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D1>, Int> {

    /**
     * Defines the dimensions of the Q-table in the format of a 3D integer array.
     *
     * The dimensions include:
     * - Row size (`rowSize`): Represents the number of rows in the Q-table.
     * - Column size (`colSize`): Represents the number of columns in the Q-table.
     * - Action size (`actionSize`): Represents the number of possible actions for each state in the Q-table.
     */
    val shape = intArrayOf(rowSize, colSize, actionSize)

    /**
     * Represents the underlying Q-table data structure for the `QTableD2` class.
     *
     * This value is initialized with a `QTableDN` instance, using the parameters `shape`, `deterministic`,
     * `tolerance`, and `defaultQValue` defined in the enclosing class. It stores the state-action value function
     * for reinforcement learning tasks, supporting multi-dimensional state spaces and multiple possible actions
     * per state.
     *
     * The `base` property serves as the core of the Q-learning model, enabling operations such as retrieving,
     * updating, and converting Q-values, and providing flexibility in representing arbitrary shaped Q-tables.
     */
    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Converts the current Q-table to a value table representation by computing the maximum Q-value
     * for each possible state. This transformation creates a V-table, which represents the optimal
     * values for all states based on the current Q-table data.
     *
     * @return A VTableD2 instance where each state corresponds to the maximum Q-value from the Q-table.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): VTableD2 {
        val Q = (if (deterministic) this else copy(true))
        var V = VTableD2(rowSize = Q.shape[0], colSize = Q.shape[1])
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state))
        }
        return V
    }

    /**
     * Retrieves the Q-value for the specified state-action pair from the Q-table.
     *
     * @param state The state represented as an `NDArray` of integers with one dimension (D1).
     * @param action The action represented as an integer.
     * @return The Q-value as a `Double` associated with the specified state and action.
     */
    override fun get(state: NDArray<Int, D1>, action: Int): Double = base[state.asDNArray(), action]

    /**
     * Retrieves the Q-value from the Q-table for a specified two-dimensional state and action.
     *
     * @param row The first dimension index of the state.
     * @param col The second dimension index of the state.
     * @param action The action represented as an integer.
     * @return The Q-value as a Double associated with the given state and action.
     */
    operator fun get(row: Int, col: Int, action: Int): Double =
        this[mk.ndarray(mk[row, col]), action]

    /**
     * Updates the Q-value in the Q-table for a specific state-action pair.
     *
     * @param state The state represented as an `NDArray` of integers with one dimension (D1).
     * @param action The action represented as an integer.
     * @param value The new Q-value to update in the table.
     * @return A new instance of `QTableD2` reflecting the updated Q-table.
     */
    override fun update(
        state: NDArray<Int, D1>,
        action: Int,
        value: Double
    ): QTableD2 =
        copy().also { it.base.table[state.toIntArray() + action] = value }

    /**
     * Updates the Q-value in the Q-table for the specified two-dimensional state and action.
     *
     * @param row The row index of the state.
     * @param col The column index of the state.
     * @param action The action represented as an integer.
     * @param value The new Q-value to update in the table.
     * @return A new instance of `QTableD2` reflecting the updated Q-table.
     */
    fun update(row: Int, col: Int, action: Int, value: Double): QTableD2 =
        update(mk.ndarray(mk[row, col]), action, value)

    /**
     * Retrieves a list of all possible states in the Q-table, represented as `NDArray` objects with one dimension (D1).
     *
     * @return A list of states, each represented as an `NDArray<Int, D1>`.
     */
    override fun allStates(): List<NDArray<Int, D1>> =
        base.allStates().map { it.asD1Array() }

    /**
     * Finds the maximum value for a given state from the Q-table.
     *
     * @param state The state represented as an `NDArray` of integers with one dimension (D1).
     * @return The maximum Q-value for the provided state, represented as a `Double`.
     */
    override fun maxValue(state: NDArray<Int, D1>): Double =
        base.maxValue(state.asDNArray())

    /**
     * Finds the maximum Q-value for a specific two-dimensional state in the Q-table.
     *
     * @param row The row index of the state.
     * @param col The column index of the state.
     * @return The maximum Q-value associated with the given state, represented as a `Double`.
     */
    fun maxValue(row: Int, col: Int): Double =
        maxValue(mk.ndarray(mk[row, col]))

    /**
     * Determines the best action to take for a given state based on the Q-table.
     *
     * @param state The current state represented as an `NDArray` of integers with one dimension (D1).
     * @return The action represented as an integer that is considered the best for the given state.
     */
    override fun bestAction(state: NDArray<Int, D1>): Int =
        base.bestAction(state.asDNArray())

    /**
     * Determines the best action for a specific state in a two-dimensional representation
     * based on the Q-table data.
     *
     * @param row The row index of the state in the Q-table.
     * @param col The column index of the state in the Q-table.
     * @return The best action for the given state, represented as an integer.
     */
    fun bestAction(row: Int, col: Int): Int =
        bestAction(mk.ndarray(mk[row, col]))

    /**
     * Creates a copy of the current `QTableD2` instance, with an optional override for its determinism property.
     *
     * @param deterministic A boolean value indicating whether the resulting copy should use deterministic behavior.
     *                       If not provided, it defaults to the current instance's `deterministic` property.
     * @return A new instance of `QTableD2`, with properties copied from the current instance.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableD2 =
        QTableD2(
            rowSize = shape[0],
            colSize = shape[1],
            actionSize = shape[2],
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the Q-table data to a file at the specified path in CSV format.
     *
     * @param path The file path where the Q-table data should be saved. The path must be a valid file system path.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads the Q-table data from a file at the specified path in CSV format and reshapes it according
     * to the current Q-table shape.
     *
     * @param path The file path from which the Q-table data should be loaded. The path must point to a
     * valid CSV file containing the Q-table data.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the underlying Q-table data to the console.
     *
     * This method provides a simple way to view the contents of the Q-table, including its
     * values and structure, for debugging or inspection purposes.
     */
    fun print() = base.print()

    /**
     * Converts the current `QTableD2` instance into a three-dimensional `QTableD3` instance with the specified shape.
     *
     * The resulting `QTableD3` instance will maintain the same data and properties, including determinism,
     * tolerance, and default Q-value, while updating the structure and dimensionality to a three-dimensional table.
     * The underlying data is copied into the new instance.
     *
     * @param rowSize The number of rows in the resulting `QTableD3`.
     * @param colSize The number of columns in the resulting `QTableD3`.
     * @param layerSize The number of layers in the resulting `QTableD3`.
     * @param actionSize The number of possible actions in the resulting `QTableD3`.
     * @return A new `QTableD3` instance with the specified dimensions and the same underlying data and properties.
     */
    fun asQTableD3(rowSize: Int,
                   colSize: Int,
                   layerSize: Int,
                   actionSize: Int): QTableD3 =
        QTableD3(
            rowSize = rowSize,
            colSize = colSize,
            layerSize = layerSize,
            actionSize = actionSize,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current `QTableD2` instance into a four-dimensional `QTableD4` instance with the specified shape.
     *
     * The resulting `QTableD4` instance retains the same data and properties, including determinism, tolerance,
     * and default Q-value, while restructuring the data into a four-dimensional table. The underlying data
     * is copied into the new instance.
     *
     * @param rowSize The number of rows in the resulting `QTableD4`.
     * @param colSize The number of columns in the resulting `QTableD4`.
     * @param layerSize The number of layers in the resulting `QTableD4`.
     * @param featureSize The number of features in the resulting `QTableD4`.
     * @param actionSize The number of possible actions in the resulting `QTableD4`.
     * @return A new `QTableD4` instance with the specified dimensions and the same underlying data and properties.
     */
    fun asQTableD4(rowSize: Int,
                   colSize: Int,
                   layerSize: Int,
                   featureSize: Int,
                   actionSize: Int): QTableD4 =
        QTableD4(
            rowSize = rowSize,
            colSize = colSize,
            layerSize = layerSize,
            featureSize = featureSize,
            actionSize = actionSize,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current object into a QTableD5 instance with the specified dimensions and configuration.
     *
     * @param rowSize The size of the rows in the QTableD5.
     * @param colSize The size of the columns in the QTableD5.
     * @param layerSize The size of the layers in the QTableD5.
     * @param featureSize The size of the features in the QTableD5.
     * @param channelSize The size of the channels in the QTableD5.
     * @param actionSize The number of actions represented in the QTableD5.
     * @return A QTableD5 instance configured with the input dimensions and properties.
     */
    fun asQTableD5(rowSize: Int,
                   colSize: Int,
                   layerSize: Int,
                   featureSize: Int,
                   channelSize: Int,
                   actionSize: Int): QTableD5 =
        QTableD5(
            rowSize = rowSize,
            colSize = colSize,
            layerSize = layerSize,
            featureSize = featureSize,
            channelSize = channelSize,
            actionSize = actionSize,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current object into a QTableDN with the provided shape and
     * copies the data from the base table into the new QTableDN instance.
     *
     * @param shape Variadic parameter representing the dimensions of the QTableDN.
     * @return A QTableDN instance with the specified shape and copied data.
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
