package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

/**
 * Represents a multi-dimensional Q-table implementation for reinforcement learning tasks with six dimensions.
 *
 * This class provides functionality to represent, update, and manipulate Q-values associated with
 * state-action pairs and supports features like determining the best action, computing maximum Q-values, and
 * exporting/importing the table data. It extends the `EnumerableQFunction` interface, allowing the representation
 * of enumerable state spaces and providing efficient state-action value updates.
 *
 * @constructor Initializes a QTableD5 object with the specified dimensions, deterministic behavior, tolerance, and
 * default Q-value for uninitialized entries.
 *
 * @param shape The dimensions of the Q-table as a vararg of integers. Must have exactly 6 entries.
 * @param deterministic A boolean flag indicating whether updates are deterministic. Defaults to true.
 * @param tolerance A double value specifying the tolerance for numerical comparisons. Defaults to 1e-6.
 * @param defaultQValue The default Q-value assigned to all state-action pairs during initialization. Defaults to 0.0.
 */
class QTableD5(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D4>, Int> {

    init {
        require(shape.size == 6) { "QTableD5 shape requires exactly 6 arguments" }
    }

    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    @Suppress("DuplicatedCode")
    override fun toV(): EnumerableValueFunction<NDArray<Int, D4>> {
        val Q = (if (deterministic) this else copy(true))
        val shape = Q.shape.dropLast(1).toIntArray()
        var V = VTableD5(shape = shape)
        for (state in allStates()) {
            V = V.update(state, Q.maxValue(state)) as VTableD5
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
    ): EnumerableQFunction<NDArray<Int, D4>, Int> =
        copy().also { it.base.table[state.toIntArray() + action] = value }


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
     * Determines the best action to take in a given state based on the Q-values.
     *
     * @param state The NDArray of type Int with 4 dimensions, representing the current state.
     * @return The integer representing the optimal action for the provided state.
     */
    override fun bestAction(state: NDArray<Int, D4>): Int =
        base.bestAction(state.asDNArray())

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
            shape = shape,
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
