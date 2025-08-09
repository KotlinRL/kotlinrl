package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * Represents a three-dimensional implementation of a Q-table for reinforcement learning.
 * It provides functionality to interact with a 3D state-action space, update Q-values,
 * retrieve the best actions or maximum Q-values for given states, and convert to higher-dimensional tables.
 *
 * @constructor Initializes a 3D Q-table with the provided shape, deterministic policy flag,
 * tolerance for non-deterministic action selection, and default Q-value.
 * @param rowSize The size of the first dimension representing the number of rows.
 * @param colSize The size of the second dimension representing the number of columns.
 * @param layerSize The size of the third dimension representing the number of layers.
 * @param actionSize The size of the forth dimension representing the number of actions.
 * @param deterministic If true, selects the best action deterministically based on maximum Q-value.
 * @param tolerance A small value for distinguishing near-equal Q-values during action selection.
 * @param defaultQValue The initial Q-value assigned to all state-action pairs.
 */
class QTableD3(
    rowSize: Int,
    colSize: Int,
    layerSize: Int,
    actionSize: Int,
    val deterministic: Boolean = true,
    val tolerance: Double = 1e-6,
    val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D2>, Int> {

    /**
     * Represents the dimensions of a multi-dimensional array or tensor.
     *
     * Each element in the array corresponds to a specific dimension:
     * - The first element represents the size of rows.
     * - The second element represents the size of columns.
     * - The third element represents the size of layers.
     * - The fourth element represents the size of actions.
     *
     * This variable is useful for defining and working with structures that
     * require multiple dimensions, such as matrices or tensors in mathematical
     * and machine learning computations.
     */
    val shape = intArrayOf(rowSize, colSize, layerSize,actionSize)

    /**
     * Represents the base Q-table used for a Q-learning algorithm.
     *
     * This variable is initialized with specified parameters such as shape,
     * determinism, tolerance, and a default Q-value. It is internal to the
     * module and used to manage state-action values in the learning process.
     */
    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Converts the current QTableD3 instance into a VTableD3 instance.
     *
     * Iterates through all possible states of the QTableD3, computes the maximum Q-value for each state,
     * and updates the VTableD3 with the corresponding values.
     *
     * If the current instance is not deterministic, creates a deterministic copy before processing.
     *
     * @return A new VTableD3 instance where each state contains the maximum Q-value from the QTableD3.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): VTableD3 {
        val Q = (if (deterministic) this else copy(true))
        var V = VTableD3(
            rowSize = Q.shape[0],
            colSize = Q.shape[1],
            layerSize = Q.shape[2],
        )
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state))
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
     * Retrieves the Q-value corresponding to the specified state and action.
     *
     * @param row The row index of the state in the three-dimensional table.
     * @param col The column index of the state in the three-dimensional table.
     * @param layer The layer index of the state in the three-dimensional table.
     * @param action The action represented as an integer.
     * @return The Q-value as a Double corresponding to the specified state and action.
     */
    operator fun get(row: Int, col: Int, layer: Int, action: Int): Double =
        this[mk.ndarray(mk[mk[row, col, layer]]), action]

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
    ): QTableD3 =
        copy().also { it.base.table[state.toIntArray() + action] = value }

    /**
     * Updates the Q-value associated with a specific state and action pair in the three-dimensional table.
     *
     * @param row The row index of the state in the table.
     * @param col The column index of the state in the table.
     * @param layer The layer index of the state in the table.
     * @param action The action represented as an integer for which the Q-value should be updated.
     * @param value The new Q-value to assign for the specified state-action pair.
     * @return A new instance of QTableD3 containing the updated Q-values.
     */
    fun update(row: Int, col: Int, layer: Int, action: Int, value: Double): QTableD3 =
        update(mk.ndarray(mk[mk[row, col, layer]]), action, value)

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
     * Determines the maximum Q-value for a specific state identified by its row, column, and layer indices
     * in a three-dimensional Q-table.
     *
     * @param row The row index of the state in the Q-table.
     * @param col The column index of the state in the Q-table.
     * @param layer The layer index of the state in the Q-table.
     * @return The maximum Q-value as a Double for the specified state.
     */
    fun maxValue(row: Int, col: Int, layer: Int): Double =
        maxValue(mk.ndarray(mk[mk[row, col, layer]]))

    /**
     * Determines the best action to take for a given state based on the Q-values.
     *
     * @param state The state represented as a 2-dimensional NDArray of integers.
     * @return The action, represented as an integer, that is determined to be the best for the specified state.
     */
    override fun bestAction(state: NDArray<Int, D2>): Int =
        base.bestAction(state.asDNArray())

    /**
     * Determines the best action to take for a given state based on its row, column, and layer indices
     * in a three-dimensional Q-table.
     *
     * @param row The row index of the state in the Q-table.
     * @param col The column index of the state in the Q-table.
     * @param layer The layer index of the state in the Q-table.
     * @return The action, represented as an integer, that is determined to be the best for the specified state.
     */
    fun bestAction(row: Int, col: Int, layer: Int): Int =
        bestAction(mk.ndarray(mk[mk[row, col, layer]]))

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
            rowSize = shape[0],
            colSize = shape[1],
            layerSize = shape[2],
            actionSize = shape[3],
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
     * Converts the current QTableD3 instance into a QTableD4 instance with the specified dimensions.
     *
     * Creates a new instance of QTableD4 and copies the underlying data, properties, and configurations
     * from the current QTableD3 instance into the resulting instance. Retains the deterministic,
     * tolerance, and defaultQValue properties of the source instance.
     *
     * @param rowSize The size of the first dimension (rows) in the new QTableD4.
     * @param colSize The size of the second dimension (columns) in the new QTableD4.
     * @param layerSize The size of the third dimension (layers) in the new QTableD4.
     * @param featureSize The size of the fourth dimension (features) in the new QTableD4.
     * @param actionSize The size of the action space for the new QTableD4.
     * @return A new QTableD4 instance with the specified dimensions and copied data.
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
     * Converts the current QTableD3 instance into a QTableD5 instance with the specified dimensions.
     *
     * Creates a new instance of QTableD5 and copies the underlying data, properties, and configurations
     * from the current QTableD3 instance into the resulting instance. Retains the deterministic,
     * tolerance, and defaultQValue properties of the source instance.
     *
     * @param rowSize The size of the first dimension (rows) in the new QTableD5.
     * @param colSize The size of the second dimension (columns) in the new QTableD5.
     * @param layerSize The size of the third dimension (layers) in the new QTableD5.
     * @param featureSize The size of the fourth dimension (features) in the new QTableD5.
     * @param channelSize The size of the fifth dimension (channels) in the new QTableD5.
     * @param actionSize The size of the action space for the new QTableD5.
     * @return A new QTableD5 instance with the specified dimensions and copied data.
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
     * Converts the current instance to a QTableDN instance with the specified shape.
     *
     * Copies the underlying Q-values, properties, and data to the new QTableDN while allowing the shape to be altered.
     * The resulting QTableDN retains the deterministic, tolerance, and defaultQValue properties from the source instance.
     *
     * @param shape The shape of the new QTableDN instance, specified as a variable number of integer dimensions.
     * @return A new QTableDN instance with the specified shape, containing the copied data and properties.
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
