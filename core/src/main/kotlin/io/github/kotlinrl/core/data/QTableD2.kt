package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * A specialized implementation of the `EnumerableQFunction` interface representing a 2-dimensional
 * Q-table for reinforcement learning tasks. This class uses a structured representation of states
 * and actions, providing functionality for managing Q-values in a deterministic or stochastic manner.
 *
 * @property shape The dimensions of the Q-table, specified as a variable number of integers. The shape
 * must explicitly contain three dimensions.
 * @property deterministic A flag indicating whether the Q-table should operate deterministically. If true,
 * the best action is selected deterministically; otherwise, stochastic behavior is applied for ties.
 * @property tolerance The allowable tolerance for determining equality of floating-point numbers, typically
 * used when comparing Q-values to resolve ties non-deterministically.
 * @property defaultQValue The initial Q-value assigned to all state-action pairs when the Q-table is instantiated.
 */
class QTableD2(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D1>, Int> {

    init {
        require(shape.size == 3) { "QTableD2 shape requires exactly 3 arguments" }
    }

    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    /**
     * Converts the current Q-table into a value function representation.
     *
     * This method calculates the value function by iterating through all states in the Q-table
     * and selecting the maximum Q-value for each state, resulting in a value function that
     * represents the expected rewards for each state under the optimal policy.
     *
     * @return A value function represented as an `EnumerableValueFunction` containing the maximum values
     *         for each state derived from the Q-table.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): EnumerableValueFunction<NDArray<Int, D1>> {
        val Q = (if (deterministic) this else copy(true))
        val shape = Q.shape.dropLast(1).toIntArray()
        var V = VTableD2(shape = shape)
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state)) as VTableD2
        }
        return V
    }

    /**
     * Retrieves the Q-value for a given state-action pair from the Q-table.
     *
     * @param state The state represented as an `NDArray` of integers with one dimension (D1).
     * @param action The action represented as an integer.
     * @return The Q-value as a `Double` corresponding to the provided state-action pair.
     */
    override fun get(state: NDArray<Int, D1>, action: Int): Double = base[state.asDNArray(), action]

    /**
     * Updates the Q-value in the Q-table for a given state-action pair with the provided value.
     *
     * @param state The state represented as an `NDArray` of integers with one dimension (D1).
     * @param action The action represented as an integer.
     * @param value The Q-value represented as a double to update the table for the provided state-action pair.
     * @return A new instance of `EnumerableQFunction` reflecting the updated Q-table.
     */
    override fun update(
        state: NDArray<Int, D1>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, D1>, Int> =
        copy().also { it.base.table[state.toIntArray() + action] = value }

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
     * Determines the best action to take for a given state based on the Q-table.
     *
     * @param state The current state represented as an `NDArray` of integers with one dimension (D1).
     * @return The action represented as an integer that is considered the best for the given state.
     */
    override fun bestAction(state: NDArray<Int, D1>): Int =
        base.bestAction(state.asDNArray())

    /**
     * Creates a copy of the current `QTableD2` instance, with an optional override for its determinism property.
     *
     * @param deterministic A boolean value indicating whether the resulting copy should use deterministic behavior.
     *                       If not provided, it defaults to the current instance's `deterministic` property.
     * @return A new instance of `QTableD2`, with properties copied from the current instance.
     */
    fun copy(deterministic: Boolean = this.deterministic): QTableD2 =
        QTableD2(
            shape = shape,
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
     * Converts the current QTableD2 instance into a QTableD3 instance with the specified shape.
     *
     * The new QTableD3 instance uses the same underlying data as the current QTableD2 but updates the
     * shape and dimensionality to match the requirements of a QTableD3. This conversion enables flexible
     * management of Q-tables with different dimensions.
     *
     * @param shape The shape of the new QTableD3 instance, which must consist of exactly 4 dimensions.
     * @return A new QTableD3 instance with the specified shape and the same underlying data and properties.
     * @throws IllegalArgumentException If the provided shape does not have exactly 4 dimensions.
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
     * Converts the current instance of `QTableD2` into a `QTableD4` with the specified shape.
     *
     * The resulting `QTableD4` instance will have the updated shape and dimensionality while preserving
     * the underlying data and properties such as determinism, tolerance, and default Q-value.
     * This method ensures that the newly created instance shares the same Q-table data as the original.
     *
     * @param shape The shape of the resulting `QTableD4`. This must consist of exactly 5 dimensions.
     * @return A new instance of `QTableD4` with the specified shape and the same underlying properties and data.
     * @throws IllegalArgumentException If the provided shape does not have exactly 5 dimensions.
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
     * Converts the current `QTableD2` instance into a `QTableD5` instance with the specified shape.
     *
     * The resulting `QTableD5` instance will maintain the same data and properties, such as determinism,
     * tolerance, and default Q-value, while updating the structure and dimensionality to a 6-dimensional Q-table.
     *
     * @param shape The shape of the new `QTableD5` instance. It must consist of exactly 6 dimensions.
     * @return A new `QTableD5` instance with the specified shape and the same underlying data and properties.
     * @throws IllegalArgumentException If the provided shape does not have exactly 6 dimensions.
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
     * Converts the current `QTableD2` instance into a `QTableDN` instance with the specified shape.
     *
     * The resulting `QTableDN` instance will inherit the same data and properties, such as determinism,
     * tolerance, and default Q-value, while allowing for an arbitrary dimensional shape.
     *
     * @param shape The shape of the new `QTableDN` instance, specified as a variable number of dimensions.
     * @return A new `QTableDN` instance with the specified shape and the same underlying data and properties.
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
