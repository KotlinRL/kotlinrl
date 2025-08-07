package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

/**
 * Represents a specialized Q-table implementation with a fixed dimensionality of 4 for the state space.
 * It facilitates the storage and manipulation of Q-values for reinforcement learning algorithms.
 *
 * @property shape Specifies the dimensions of the state-action space, requiring exactly 5 arguments where the
 *     last dimension corresponds to the action space.
 * @property deterministic Determines whether the best action is selected in a deterministic manner or stochastically
 *     based on tolerance.
 * @property tolerance Defines the threshold for considering Q-values as equal when selecting one among
 *     multiple best actions in a non-deterministic mode.
 * @property defaultQValue Specifies the initialized Q-value for all state-action pairs in the table.
 */
class QTableD4(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D3>, Int> {

    init {
        require(shape.size == 5) { "QTableD4 shape requires exactly 5 arguments" }
    }

    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Converts the Q-table to a value function representation.
     *
     * This method generates an enumerable value function by iterating over all states in the Q-table
     * and calculating the maximum Q-value for each state. The resulting value function is returned
     * as an instance of `EnumerableValueFunction<NDArray<Int, D3>>`.
     *
     * @return An enumerable value function representing the maximum Q-values for each state.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): EnumerableValueFunction<NDArray<Int, D3>> {
        val Q = (if (deterministic) this else copy(true))
        val shape = Q.shape.dropLast(1).toIntArray()
        var V = VTableD4(shape = shape)
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state)) as VTableD4
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
    ): EnumerableQFunction<NDArray<Int, D3>, Int> =
        copy().also { it.base.table[state.toIntArray() + action] = value }


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
     * Creates a copy of the current `QTableD4` instance with the option to modify its determinism.
     *
     * @param deterministic Specifies whether the copied instance should use a deterministic approach.
     *                       Defaults to the value of `this.deterministic` in the current instance.
     * @return A new `QTableD4` instance with the same properties as the current instance,
     *         but with the specified determinism setting.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableD4 =
        QTableD4(
            shape = shape,
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
     * This method reshapes the Q-table into a higher-dimensional representation
     * with support for a 5-dimensional shape while retaining other configuration properties
     * like determinism, tolerance, and default Q-value. The internal data of the
     * QTableD4 is copied into the new QTableD5.
     *
     * @param shape A variadic parameter that specifies the dimensions for the resulting QTableD5.
     * Each value represents the size of a corresponding dimension.
     * @return A new instance of QTableD5 with the specified shape and copied data from the current QTableD4.
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
