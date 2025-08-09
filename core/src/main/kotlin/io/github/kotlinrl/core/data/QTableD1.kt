package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Represents a one-dimensional Q-table used in reinforcement learning environments to map
 * state-action pairs to Q-values. The table supports deterministic and non-deterministic modes
 * for handling state-action Q-values.
 *
 * @param stateSize The number of states in the one-dimensional Q-table.
 * @param actionSize The number of actions for each state in the Q-table.
 * @param deterministic Indicates whether the Q-table operates in deterministic mode. Defaults to true.
 * @param tolerance The tolerance used for handling floating-point precision comparisons. Defaults to 1e-6.
 * @param defaultQValue The initial default Q-value assigned to all state-action pairs. Defaults to 0.0.
 */
class QTableD1(
    stateSize: Int,
    actionSize: Int,
    val deterministic: Boolean = true,
    val tolerance: Double = 1e-6,
    val defaultQValue: Double = 0.0
) : EnumerableQFunction<Int, Int> {
    /**
     * Represents the shape of the underlying Q-table in the `QTableD1` instance.
     * It is defined as a two-element array, where:
     * - The first element corresponds to the number of states (`stateSize`).
     * - The second element corresponds to the number of actions (`actionSize`).
     */
    val shape = intArrayOf(stateSize, actionSize)

    /**
     * The base Q-table implementation for the `QTableD1` class.
     * This property is an instance of `QTableDN` that holds the core Q-values and provides
     * functionality for managing the underlying table, including shaping, determinism,
     * tolerance levels, and default Q-value initialization.
     *
     * It serves as the foundational data structure for the `QTableD1` class, providing storage
     * and manipulation of Q-values across states and actions.
     */
    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Converts the current QTableD1 instance into a VTableD1 by extracting the maximum Q-value for each state.
     * This process yields a value function representation based on the state space.
     *
     * @return A VTableD1 instance representing the value function computed from the Q-values.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): VTableD1 {
        val Q = if (this.deterministic) this else copy(true)
        val shape = Q.shape.dropLast(1).toIntArray()
        var V = VTableD1(shape[0])
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state)) as VTableD1
        }
        return V
    }

    /**
     * Retrieves the Q-value for the specified state and action from the Q-table.
     *
     * @param state An integer representing the state for which the Q-value is to be retrieved.
     * @param action An integer representing the action for which the Q-value is to be retrieved.
     * @return The Q-value corresponding to the specified state and action.
     */
    override fun get(state: Int, action: Int): Double =
        base[mk.ndarray(mk[state]).asDNArray(), action]

    /**
     * Updates the Q-value for a specific state and action in the QTableD1.
     *
     * @param state The integer index representing the state to update.
     * @param action The integer index representing the action to update.
     * @param value The new Q-value to set for the specified state and action.
     * @return A new QTableD1 instance with the updated Q-value.
     */
    override fun update(
        state: Int,
        action: Int,
        value: Double
    ): QTableD1 =
        copy().also { it.base.table[intArrayOf(state) + action] = value }

    /**
     * Retrieves a list of all states in the Q-table, where each state is extracted
     * using the first element of its corresponding index representation.
     *
     * @return A list of integers representing the states within the Q-table.
     */
    override fun allStates(): List<Int> =
        base.allStates().map { it[0] }

    /**
     * Retrieves the maximum Q-value for a specific state in the Q-table.
     *
     * @param state An integer representing the state for which the maximum Q-value is to be determined.
     * @return The maximum Q-value as a double for the specified state.
     */
    override fun maxValue(state: Int): Double =
        base.maxValue(mk.ndarray(intArrayOf(state)).asDNArray())

    /**
     * Determines the best action to take for a given state based on the Q-values.
     * This method delegates the action selection process to the base Q-table framework.
     *
     * @param state An integer representing the state for which the best action is to be determined.
     * @return An integer representing the best action for the given state.
     */
    override fun bestAction(state: Int): Int =
        base.bestAction(mk.ndarray(intArrayOf(state)).asDNArray())

    /**
     * Creates a copy of the current QTableD1 instance, with an option to specify whether the new instance
     * should use a deterministic strategy.
     *
     * @param deterministic A boolean indicating whether the new QTableD1 should be deterministic.
     *                       Defaults to the `deterministic` property of the current instance.
     * @return A new QTableD1 instance with the same attributes and data as the current instance,
     *         but with the specified `deterministic` setting.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableD1 =
        QTableD1(
            stateSize = shape[0],
            actionSize = shape[1],
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
     * Converts the current QTableD1 instance into a QTableD2 instance with the specified dimensions (rowSize, colSize, actionSize).
     * This includes copying the underlying data and attributes from the original QTableD1 instance to the new QTableD2 instance.
     *
     * @param rowSize The number of rows in the resulting QTableD2.
     * @param colSize The number of columns in the resulting QTableD2.
     * @param actionSize The number of actions in the resulting QTableD2.
     * @return A new QTableD2 instance with the specified dimensions, initialized with the Q-values and attributes of the current QTableD1 instance.
     */
    fun asQTableD2(rowSize: Int, colSize: Int, actionSize: Int,): QTableD2 =
        QTableD2(
            rowSize = rowSize,
            colSize = colSize,
            actionSize = actionSize,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Converts the current QTableD1 instance into a QTableD3 instance with the specified dimensions (colSize, rowSize, layerSize, actionSize).
     * This includes copying the underlying data and attributes from the original QTableD1 instance to the new QTableD3 instance.
     *
     * @param colSize The number of columns in the resulting QTableD3.
     * @param rowSize The number of rows in the resulting QTableD3.
     * @param layerSize The number of layers in the resulting QTableD3.
     * @param actionSize The number of actions in the resulting QTableD3.
     * @return A new QTableD3 instance with the specified dimensions, initialized with the Q-values and attributes of the current QTableD1 instance.
     */
    fun asQTableD3(colSize: Int, rowSize: Int, layerSize: Int, actionSize: Int,): QTableD3 =
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
     * Converts the current QTableD1 instance into a QTableD4 instance with the specified dimensions.
     * This includes copying the underlying data and attributes from the original QTableD1 instance to the new QTableD4 instance.
     *
     * @param rowSize The size of the rows in the resulting QTableD4.
     * @param colSize The size of the columns in the resulting QTableD4.
     * @param layerSize The number of layers in the resulting QTableD4.
     * @param featureSize The number of features in each layer of the resulting QTableD4.
     * @param actionSize The number of possible actions in the resulting QTableD4.
     * @return A new QTableD4 instance with the specified dimensions, initialized with the Q-values and attributes of the current QTableD1 instance.
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
     * Converts the current QTableD1 instance into a QTableD5 instance with the specified dimensions.
     * This involves copying the underlying data and attributes from the original QTableD1 instance
     * to the new QTableD5 instance.
     *
     * @param rowSize The number of rows in the resulting QTableD5.
     * @param colSize The number of columns in the resulting QTableD5.
     * @param layerSize The number of layers in the resulting QTableD5.
     * @param featureSize The number of features in each layer of the resulting QTableD5.
     * @param channelSize The number of channels in the resulting QTableD5.
     * @param actionSize The number of actions in the resulting QTableD5.
     * @return A new QTableD5 instance initialized with the specified dimensions and the Q-values
     * from the current QTableD1 instance.
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
     * Converts the current QTableD1 instance into a QTableDN instance with the specified shape dimensions.
     * The data from the original Q-table is copied into the new instance, preserving the original Q-values.
     *
     * @param shape The dimensions of the resulting QTableDN. Each integer in this variable-length argument
     *              specifies the size of the corresponding dimension.
     * @return A new QTableDN instance initialized with the specified dimensions and Q-values from the current instance.
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
