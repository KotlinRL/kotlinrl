package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * A 4-dimensional implementation of a Q-function that maps state-action pairs to Q-values.
 * The Q-table is represented as a hypercube with 5 total dimensions, the last being the action dimension.
 * This enables efficient mapping of states with 3 dimensions and a discrete set of actions to corresponding Q-values.
 *
 * @param deterministic If true, a deterministic strategy is used. Default is true.
 * @param tolerance Specifies the tolerance level for floating-point operations when comparing Q-values. Default is 1e-6.
 * @param defaultQValue The default Q-value assigned to state-action pairs when initializing the Q-table. Default is 0.0.
 */
class QTableD4(
    rowSize: Int,
    colSize: Int,
    layerSize: Int,
    featureSize: Int,
    actionSize: Int,
    val deterministic: Boolean = true,
    val tolerance: Double = 1e-6,
    val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D3>, Int> {

    /**
     * Represents the multidimensional shape of the Q-table in `QTableD4`.
     *
     * The shape is defined as a 5-dimensional array with the following components:
     * - `rowSize`: the number of rows in the state representation.
     * - `colSize`: the number of columns in the state representation.
     * - `layerSize`: the number of layers in the state representation.
     * - `featureSize`: the number of features in the state representation.
     * - `actionSize`: the number of possible actions for each state.
     *
     * This variable helps determine the structure and indexing of the Q-table.
     */
    val shape = intArrayOf(rowSize, colSize, layerSize, featureSize, actionSize)

    /**
     * Represents the underlying multi-dimensional Q-table used for storing Q-values in the `QTableD4` class.
     * This Q-table is initialized with the provided shape, determinism setting, tolerance, and default Q-value.
     * It serves as the core data structure for reinforcement learning operations, including value retrieval,
     * updates, and conversions.
     */
    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Transforms the current Q-table into a value table by computing the maximum Q-value for each state.
     *
     * The transformation involves iterating through all possible states and retaining only the
     * maximum Q-value associated with each state, effectively converting the Q-table to a table
     * of state values.
     *
     * @return A new `VTableD4` instance representing the value table, where each entry corresponds
     * to the maximum Q-value for the respective state.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): VTableD4 {
        val Q = (if (deterministic) this else copy(true))
        var V = VTableD4(
            rowSize = Q.shape[0],
            colSize = Q.shape[1],
            layerSize = Q.shape[2],
            featureSize = Q.shape[3],
        )
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state))
        }
        return V
    }

    /**
     * Retrieves the Q-value associated with a given state and action.
     *
     * @param state The state represented as an NDArray of type Int with 3 dimensions.
     * @param action The action represented as an integer.
     * @return The Q-value corresponding to the given state and action as a Double.
     */
    override fun get(state: NDArray<Int, D3>, action: Int): Double = base[state.asDNArray(), action]

    /**
     * Retrieves the Q-value for a specific combination of row, column, layer, feature, and action
     * from the Q-table.
     *
     * @param row The row index representing a specific dimension of the state.
     * @param col The column index representing another dimension of the state.
     * @param layer The layer index representing the third dimension of the state.
     * @param feature The feature index representing the fourth dimension of the state.
     * @param action The action index representing the action taken in this state.
     * @return The Q-value corresponding to the specified state and action as a Double.
     */
    operator fun get(row: Int, col: Int, layer: Int, feature: Int, action: Int): Double  =
        this[mk.ndarray(mk[mk[mk[row, col, layer, feature]]]), action]

    /**
     * Updates the Q-value for a given state and action in the Q-table.
     *
     * @param state The current state represented as an NDArray of type Int with 3 dimensions.
     * @param action The action represented as an integer.
     * @param value The new Q-value to be assigned to the specified state-action pair.
     * @return A new instance of EnumerableQFunction with the updated Q-table.
     */
    override fun update(
        state: NDArray<Int, D3>,
        action: Int,
        value: Double
    ): QTableD4 =
        copy().also { it.base.table[state.toIntArray() + action] = value }

    /**
     * Updates the Q-value for a specific combination of row, column, layer, feature, and action in the Q-table.
     *
     * @param row The row index representing a specific dimension of the state.
     * @param col The column index representing another dimension of the state.
     * @param layer The layer index representing the third dimension of the state.
     * @param feature The feature index representing the fourth dimension of the state.
     * @param action The action index representing the action taken in this state.
     * @param value The new Q-value to be assigned to the specified state-action pair.
     * @return A new instance of QTableD4 with the updated Q-table.
     */
    fun update(row: Int, col: Int, layer: Int, feature: Int, action: Int, value: Double): QTableD4 =
        update(mk.ndarray(mk[mk[mk[row, col, layer, feature]]]), action, value)

    /**
     * Retrieves all possible states from the Q-table, transforming them into NDArray objects
     * with a fixed dimensionality of 3 (D3 arrays).
     *
     * @return A list of NDArray objects, each representing a state in 3 dimensions.
     */
    override fun allStates(): List<NDArray<Int, D3>> =
        base.allStates().map { it.asD3Array() }

    /**
     * Determines the maximum Q-value associated with a given state.
     *
     * @param state The state represented as an NDArray of type Int with 3 dimensions.
     * @return The maximum Q-value corresponding to the given state as a Double.
     */
    override fun maxValue(state: NDArray<Int, D3>): Double =
        base.maxValue(state.asDNArray())

    /**
     * Determines the maximum Q-value for a specific state represented by its indices.
     *
     * @param row The row index representing a specific dimension of the state.
     * @param col The column index representing another dimension of the state.
     * @param layer The layer index representing the third dimension of the state.
     * @param feature The feature index representing the fourth dimension of the state.
     * @return The maximum Q-value corresponding to the specified state as a Double.
     */
    fun maxValue(row: Int, col: Int, layer: Int, feature: Int): Double =
        maxValue(mk.ndarray(mk[mk[mk[row, col, layer, feature]]]))

    /**
     * Determines the best action to take for a given state based on the Q-values.
     *
     * @param state The state represented as an NDArray of type Int with 3 dimensions.
     *              It represents the current state of the environment.
     * @return The best action as an integer. The action is determined based on either a deterministic
     *         or stochastic strategy, depending on the configuration.
     */
    override fun bestAction(state: NDArray<Int, D3>): Int =
        base.bestAction(state.asDNArray())

    /**
     * Determines the best action to take for a specific state in the Q-table using row, column, layer, and feature indices.
     *
     * @param row The row index representing a specific dimension of the state.
     * @param col The column index representing another dimension of the state.
     * @param layer The layer index representing the third dimension of the state.
     * @param feature The feature index representing the fourth dimension of the state.
     * @return The best action as an integer, determined based on either a deterministic or stochastic strategy.
     */
    fun bestAction(row: Int, col: Int, layer: Int, feature: Int): Int =
        bestAction(mk.ndarray(mk[mk[mk[row, col, layer, feature]]]))

    /**
     * Creates a copy of the current `QTableD4` instance with the option to modify its determinism.
     *
     * @param deterministic Specifies whether the copied instance should use a deterministic approach.
     *                       Defaults to the value of `this.deterministic` in the current instance.
     * @return A new `QTableD4` instance with the same properties as the current instance,
     *         but with the specified determinism setting.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableD4 =
        QTableD4(
            rowSize = shape[0],
            colSize = shape[1],
            layerSize = shape[2],
            featureSize = shape[3],
            actionSize = shape[4],
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the Q-table to a file at the specified path in CSV format.
     *
     * @param path The file path where the Q-table will be saved. The path should include the
     * file name and extension.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads a Q-table from a CSV file located at the specified path and reshapes it based on the table's dimensions.
     *
     * @param path The file path to the CSV file containing the Q-table data. The path should include the file name and extension.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the internal representation of the Q-table to the standard output.
     *
     * This method outputs the contents of the Q-table for inspection or debugging purposes.
     * The exact output format depends on the implementation of the `base.print()` function.
     */
    fun print() = base.print()

    /**
     * Converts the current QTableD4 instance into a QTableD5 instance.
     * This method reshapes the Q-table into a five-dimensional representation
     * while retaining configuration properties such as determinism, tolerance,
     * and default Q-value. The internal data of the current QTableD4 instance
     * is copied into the new QTableD5.
     *
     * @param rowSize The size of the first dimension (rows) in the resulting QTableD5.
     * @param colSize The size of the second dimension (columns) in the resulting QTableD5.
     * @param layerSize The size of the third dimension (layers) in the resulting QTableD5.
     * @param featureSize The size of the fourth dimension (features) in the resulting QTableD5.
     * @param channelSize The size of the fifth dimension (channels) in the resulting QTableD5.
     * @param actionSize The size of the last dimension (actions) in the resulting QTableD5.
     * @return A new QTableD5 instance with the specified dimensions and the copied data from the current QTableD4.
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
     * Converts the current QTableD4 instance into a QTableDN instance.
     * This method reshapes the Q-table to the specified multi-dimensional representation
     * while retaining configuration properties like determinism, tolerance, and default Q-value.
     * The internal data of the current QTableD4 is copied into the new QTableDN.
     *
     * @param shape A variadic parameter that specifies the dimensions for the resulting QTableDN.
     * Each value corresponds to the size of a particular dimension. The shape must contain at least two dimensions.
     * @return A new QTableDN instance with the specified shape and the copied data from the current QTableD4.
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
