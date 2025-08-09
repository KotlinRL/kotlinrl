package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * Represents a specific implementation of an enumerable Q-function for a Q-table
 * with a 5-dimensional state-action space. It provides support for operations such as
 * retrieving Q-values, updating Q-values, enumerating all possible states, and
 * exporting to other related structures like value tables.
 *
 * @constructor
 * Initializes a new instance of `QTableD5` with the specified shape, determinism,
 * tolerance, and default Q-value.
 *
 * @param deterministic Indicates whether deterministic updates are enabled. Defaults to `true`.
 * @param tolerance Specifies the tolerance level for numerical calculations. Defaults to `1e-6`.
 * @param defaultQValue The initial default Q-value for all state-action pairs. Defaults to `0.0`.
 */
class QTableD5(
    rowSize: Int,
    colSize: Int,
    layerSize: Int,
    featureSize: Int,
    channelSize: Int,
    actionSize: Int,
    val deterministic: Boolean = true,
    val tolerance: Double = 1e-6,
    val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D4>, Int> {

    /**
     * Represents the dimensional structure of the Q-table in a 6D space.
     *
     * The array defines the size of each dimension:
     * - `rowSize`: The number of rows in the Q-table.
     * - `colSize`: The number of columns in the Q-table.
     * - `layerSize`: The number of layers in the Q-table.
     * - `featureSize`: The number of features in the Q-table.
     * - `channelSize`: The number of channels in the Q-table.
     * - `actionSize`: The number of actions available per state.
     */
    val shape = intArrayOf(rowSize, colSize, layerSize, featureSize, channelSize, actionSize)

    /**
     * Represents the core data structure used for storing Q-values in the context of
     * the QTableD5 class. This property is initialized with specific parameters
     * including the shape, determinism, tolerance, and a default Q-value.
     *
     * Acts as the internal storage mechanism for Q-values, enabling operations
     * such as querying, updating, and managing Q-values in a multi-dimensional
     * state-action space.
     *
     * This property is immutable and encapsulates all fundamental behaviors
     * necessary for Q-value management within the QTableD5 class.
     */
    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Converts the current QTableD5 instance to a VTableD5 object by computing the maximum
     * Q-value for all possible states and storing these values in a new VTableD5 instance.
     *
     * @return A VTableD5 instance where each state corresponds to the maximum Q-value
     *         from the current QTableD5 instance.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): VTableD5 {
        val Q = (if (deterministic) this else copy(true))
        var V = VTableD5(
            rowSize = Q.shape[0],
            colSize = Q.shape[1],
            layerSize = Q.shape[2],
            featureSize = Q.shape[3],
            channelSize = Q.shape[4])
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state))
        }
        return V
    }

    /**
     * Retrieves the Q-value for a given state and action.
     *
     * @param state The NDArray of type Int with 4 dimensions, representing the current state.
     * @param action The integer representing the action to be evaluated in the given state.
     * @return The Q-value as a Double corresponding to the provided state and action.
     */
    override fun get(state: NDArray<Int, D4>, action: Int): Double = base[state.asDNArray(), action]

    /**
     * Retrieves the Q-value corresponding to the specified state and action.
     *
     * @param row The row index of the state.
     * @param col The column index of the state.
     * @param layer The layer index of the state.
     * @param feature The feature index of the state.
     * @param channel The channel index of the state.
     * @param action The action index for which the Q-value is requested.
     * @return The Q-value as a Double for the specified state and action.
     */
    operator fun get(row: Int, col: Int, layer: Int, feature: Int, channel: Int, action: Int): Double =
        this[mk.ndarray(mk[mk[mk[mk[row, col, layer, feature, channel]]]]), action]

    /**
     * Updates the Q-value for a given state and action with the specified value.
     *
     * @param state The NDArray of type Int with 4 dimensions, representing the current state.
     * @param action The integer representing the action to be updated for the given state.
     * @param value The double value to set as the Q-value for the specified state and action.
     * @return A new instance of EnumerableQFunction with the updated Q-value applied.
     */
    override fun update(
        state: NDArray<Int, D4>,
        action: Int,
        value: Double
    ): QTableD5 =
        copy().also { it.base.table[state.toIntArray() + action] = value }

    /**
     * Updates the Q-value for a specific state and action with the given value.
     *
     * @param row The row index of the state.
     * @param col The column index of the state.
     * @param layer The layer index of the state.
     * @param feature The feature index of the state.
     * @param channel The channel index of the state.
     * @param action The action index for which the Q-value is updated.
     * @param value The new Q-value to be set for the specified state and action.
     * @return A new instance of QTableD5 with the updated Q-value applied.
     */
    fun update(row: Int, col: Int, layer: Int, feature: Int, channel: Int, action: Int, value: Double): QTableD5 =
        update(mk.ndarray(mk[mk[mk[mk[row, col, layer, feature, channel]]]]), action, value)

    /**
     * Retrieves all possible states as a list of 4-dimensional NDArrays.
     *
     * This method maps the states from a base representation into a list of NDArrays
     * with a shape corresponding to 4 dimensions.
     *
     * @return A list of NDArray instances of type Int and dimension D4, representing all possible states.
     */
    override fun allStates(): List<NDArray<Int, D4>> =
        base.allStates().map { it.asD4Array() }

    /**
     * Computes the maximum Q-value for a given state.
     *
     * @param state The NDArray of type Int with 4 dimensions, representing the current state.
     * @return The maximum Q-value as a Double for the given state.
     */
    override fun maxValue(state: NDArray<Int, D4>): Double =
        base.maxValue(state.asDNArray())

    /**
     * Computes the maximum Q-value for the specified state parameters.
     *
     * @param row The row index of the state.
     * @param col The column index of the state.
     * @param layer The layer index of the state.
     * @param feature The feature index of the state.
     * @param channel The channel index of the state.
     * @return The maximum Q-value as a Double for the specified state.
     */
    fun maxValue(row: Int, col: Int, layer: Int, feature: Int, channel: Int): Double =
        maxValue(mk.ndarray(mk[mk[mk[mk[row, col, layer, feature, channel]]]]))

    /**
     * Determines the best action to take in a given state based on the Q-values.
     *
     * @param state The NDArray of type Int with 4 dimensions, representing the current state.
     * @return The integer representing the optimal action for the provided state.
     */
    override fun bestAction(state: NDArray<Int, D4>): Int =
        base.bestAction(state.asDNArray())

    /**
     * Determines the best action to take for a specific state, defined by the given indices, based on the Q-values.
     *
     * @param row The row index of the state.
     * @param col The column index of the state.
     * @param layer The layer index of the state.
     * @param feature The feature index of the state.
     * @param channel The channel index of the state.
     * @return The integer representing the optimal action for the specified state.
     */
    fun bestAction(row: Int, col: Int, layer: Int, feature: Int, channel: Int): Int =
        bestAction(mk.ndarray(mk[mk[mk[mk[row, col, layer, feature, channel]]]])).toInt()

    /**
     * Creates a copy of the current QTableD5 instance, optionally overriding the `deterministic` flag.
     *
     * @param deterministic A Boolean value indicating whether the copied instance should use
     *                      deterministic updates. Defaults to the current instance's `deterministic` value.
     * @return A new QTableD5 instance with the same properties as the current instance but with the updated
     *         deterministic configuration if specified.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableD5 =
        QTableD5(
            rowSize = shape[0],
            colSize = shape[1],
            layerSize = shape[2],
            featureSize = shape[3],
            channelSize = shape[4],
            actionSize = shape[5],
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    /**
     * Saves the current QTableD5 instance to a file at the specified path.
     *
     * @param path The file system path as a String where the QTableD5 instance will be saved.
     */
    fun save(path: String) = base.save(path)

    /**
     * Loads data from the specified file path and updates the internal data structure of the QTableD5 instance.
     *
     * @param path The file system path as a String from which the QTableD5 data will be loaded.
     */
    fun load(path: String) = base.load(path)

    /**
     * Prints the current representation of the underlying data or object.
     *
     * This function delegates the print functionality to the `base` property
     * of the enclosing class, relying on `base.print()` to output content.
     */
    fun print() = base.print()

    /**
     * Creates a new instance of QTableDN with the same characteristics as this QTableD5 instance,
     * but with a shape defined by the provided argument. Copies over the data from the current
     * table and applies it to the new QTableDN instance.
     *
     * @param shape The shape for the new QTableDN instance as a vararg of integers, specifying
     *              the dimensions and size of the Q-table.
     * @return A newly initialized QTableDN object with the specified shape and inherited settings.
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
