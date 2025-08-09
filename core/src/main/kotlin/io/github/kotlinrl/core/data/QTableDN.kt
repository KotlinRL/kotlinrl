package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

/**
 * A Q-table representation for reinforcement learning with support for multidimensional states
 * using `NDArray` data structures. The `QTableDN` class provides methods for managing state-action
 * relationships, calculating optimal actions, and updating the `Q-values`, while supporting both deterministic
 * and non-deterministic behavior.
 *
 * @property shape The dimensions of the Q-table, where the last dimension represents the action space.
 * @property deterministic Indicates whether the `bestAction` method selects the action deterministically or
 * considers probabilistic action selection with tolerance.
 * @property tolerance A tolerance value used when selecting actions probabilistically. It determines how similar
 * Q-values should be considered when choosing among near-optimal actions in non-deterministic mode.
 * @property defaultQValue The default value assigned to Q-value entries in the table when initialized.
 */
class QTableDN(
    vararg val shape: Int,
    val deterministic: Boolean = true,
    val tolerance: Double = 1e-6,
    val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, DN>, Int> {

    init {
        require(shape.size >= 2) { "QTableDN shape requires at least 2 arguments" }
    }

    /**
     * Represents a multidimensional array used to store Q-values for state-action pairs in the Q-learning algorithm.
     * The array is initialized with default Q-values and follows a shape defined by the state and action dimensions.
     *
     * This variable is internal to the `QTableDN` class and is accessed and modified through various methods
     * to perform operations such as retrieving Q-values, updating specific entries, and converting the Q-table.
     */
    internal var table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { defaultQValue }.asDNArray()

    /**
     * Converts the current QTableDN instance into a VTableDN instance by computing the maximum
     * Q-value for each state across all possible actions.
     *
     * @return A new VTableDN instance where each state's value is defined as the maximum Q-value
     *         for that state in the current Q-table.
     */
    @Suppress("DuplicatedCode")
    override fun toV(): VTableDN {
        val Q = (if (deterministic) this else copy(true))
        val shape = Q.shape.dropLast(1).toIntArray()
        var V = VTableDN(shape = shape)
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state))
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
     * Retrieves an element from the internal Q-table based on the given state-action pair.
     *
     * @param stateAction A variable number of integers representing the state-action combination
     *                    for which the Q-value needs to be fetched.
     * @return The value associated with the specified state-action pair in the Q-table.
     */
    operator fun get(vararg stateAction: Int) = table[stateAction]

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
    ): QTableDN =
        copy().also { it.table[state.toIntArray() + action] = value }

    /**
     * Retrieves a list of all possible states within the multidimensional structure defined
     * by the Q-table's dimensions, excluding the action dimension.
     *
     * @return A list of states, where each state is represented as an NDArray of integers.
     */
    override fun allStates(): List<NDArray<Int, DN>> {
        val stateShape = shape.dropLast(1).toIntArray()
        val stateRank = stateShape.size
        val allStates = mutableSetOf<NDArray<Int, DN>>()

        for (index in table.multiIndices) {
            val stateIndex = index.dropLast(1).toIntArray()
            allStates += toNestedNDArray(stateIndex, stateRank)
        }

        return allStates.toList()
    }

    /**
     * Computes the Q-values for a given state from the Q-table.
     *
     * @param state The state represented as an NDArray of integers.
     * @return An NDArray of doubles containing the Q-values for the given state.
     */
    private fun qValues(state: NDArray<Int, DN>): NDArray<Double, D1> {
        val idx = state.toIntArray()              // length should be stateRank (4 for QTableD5)
        val stateRank = shape.size - 1            // last dim is actions

        require(idx.size == stateRank) {
            "State rank mismatch: got ${idx.size} indices but Q has $stateRank state dims. Q.shape=${shape.contentToString()}"
        }

        // Iteratively index each state axis; after each view, remaining state axes shift to axis 0
        var slice = table
        for (i in 0 until stateRank) {
            slice = slice.view(index = idx[i], axis = 0).asDNArray()
        }

        // Now only the action dim should remain
        require(slice.dim.d == 1) {
            "Expected 1D action vector, got ${slice.dim.d}D with shape ${slice.shape.contentToString()}"
        }
        return slice.asDNArray().asD1Array()
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
}