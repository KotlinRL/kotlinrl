package io.github.kotlinrl.core

import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Type alias for `QTableD1`, representing a one-dimensional Q-value table for reinforcement learning.
 * Encapsulates Q-value mappings for state-action pairs, with support for various operations like
 * retrieving Q-values, updating them, and transforming the table into different dimensions or formats.
 */
typealias QTableD1 = io.github.kotlinrl.core.data.QTableD1
/**
 * Alias for the `QTableD2` class, which represents a specific implementation of a
 * two-dimensional Q-Table for reinforcement learning.
 *
 * `QTableD2` is structured with three required dimensions for state-action representation
 * and includes parameters to control determinism, tolerance, and default Q-values.
 *
 * This alias provides a more concise way to reference the class within the codebase.
 */
typealias QTableD2 = io.github.kotlinrl.core.data.QTableD2
/**
 * A type alias for the `QTableD3` class, which represents a 3-dimensional Q-table.
 * This is used in reinforcement learning for storing and managing Q-values
 * for state-action pairs in a deterministic or non-deterministic environment.
 */
typealias QTableD3 = io.github.kotlinrl.core.data.QTableD3
/**
 * A type alias for `QTableD4` class.
 *
 * Represents a Q-function implemented using a 4-dimensional data structure to
 * store state-action values (Q-values). It supports operations such as state-value retrieval,
 * updates, saving/loading, and conversions to other QTable formats.
 *
 * `QTableD4` operates on states represented as `NDArray<Int, D3>` and actions as integers.
 * It is deterministic by default, but allows for configuration of internal parameters
 * such as tolerance and default Q-value.
 */
typealias QTableD4 = io.github.kotlinrl.core.data.QTableD4
/**
 * A type alias for the `QTableDN` class, which provides a tabular representation of a Q-function
 * for reinforcement learning. The Q-table is designed to work with multi-dimensional state and action spaces
 * using N-dimensional arrays.
 */
typealias QTableDN = io.github.kotlinrl.core.data.QTableDN
/**
 * A type alias for the VTableD1 class, which represents a one-dimensional value table
 * designed for reinforcement learning or other computational use-cases. The VTableD1
 * requires a single integer shape definition. It provides functionalities such as
 * state-value retrieval, state-value updates, and conversions to higher-dimensional
 * value tables, as well as saving and loading table data.
 *
 * This type alias simplifies access to the VTableD1 class from the io.github.kotlinrl.core.data package.
 */
typealias VTableD1 = io.github.kotlinrl.core.data.VTableD1
/**
 * A type alias for the VTableD2 class, representing a two-dimensional value table.
 * This alias provides a more concise reference to the VTableD2 class within the library.
 */
typealias VTableD2 = io.github.kotlinrl.core.data.VTableD2
/**
 * Type alias for `VTableD3`, a specific implementation of a value table for 3-dimensional problems.
 *
 * `VTableD3` allows efficient manipulation and storage of values associated with enumerable states
 * in a 3-dimensional grid. It provides core functionality for accessing, updating, and transforming
 * value functions and is useful in reinforcement learning and other domains requiring structured
 * state-value mappings.
 */
typealias VTableD3 = io.github.kotlinrl.core.data.VTableD3
/**
 * Typealias for `io.github.kotlinrl.core.data.VTableD4`, representing a 4-dimensional value table.
 * This is a specialized implementation of an enumerable value function for managing
 * states and values in 4 dimensions.
 */
typealias VTableD4 = io.github.kotlinrl.core.data.VTableD4
/**
 * Type alias for the `VTableDN` class, which represents a value table with n-dimensional
 * state space. It operates as an enumerable value function designed for use in reinforcement
 * learning scenarios.
 *
 * The `VTableDN` class allows defining and manipulating tables storing values associated
 * with states represented as n-dimensional arrays. It provides features like value retrieval,
 * updates, and methods to obtain all possible states.
 */
typealias VTableDN = io.github.kotlinrl.core.data.VTableDN

/**
 * Represents a composite key consisting of a state and an action, primarily designed for use
 * within environments, such as those in reinforcement learning or simulation contexts.
 * This key enables the pairing of states and actions in a structured manner and provides
 * comparison functionality to support consistent ordering.
 *
 * The comparison logic is defined based on the `Comparable` representations of the `state`
 * and `action` components. Both components are converted to their comparable forms using
 * the `toComparable` extension function. The state is compared first, and if the states
 * are identical, the comparison defers to the action.
 *
 * @param State The type representing the state in the key. It must be convertible to a
 * comparable format.
 * @param Action The type representing the action in the key. It must be convertible to a
 * comparable format.
 */
data class StateActionKey<State, Action>(
    val state: State,
    val action: Action
) : Comparable<StateActionKey<State, Action>> {
    private val nnState: Comparable<State>
    private val nnAction: Comparable<Action>

    init {
        require(state != null) { "State cannot be null." }
        require(action != null) { "Action cannot be null." }
        nnState = state.toComparable()
        nnAction = action.toComparable()
    }
    override fun compareTo(other: StateActionKey<State, Action>): Int {
        val stateCmp = nnState.compareTo(other.state)
        return if (stateCmp != 0) stateCmp else nnAction.compareTo(other.action)
    }
}

/**
 * Represents a wrapper around an `NDArray` object that allows for comparisons
 * between instances based on their data content and size.
 *
 * The class encapsulates an `NDArray` of integers and provides functionality
 * to compare two `ComparableNDArray` objects lexicographically. The comparison
 * is first based on the size of the arrays, and then element-wise if the sizes
 * are equal. The class also provides a way to retrieve the underlying `NDArray`
 * and customizes the string representation to directly reflect the `NDArray`.
 *
 * This class implements Kotlin's `Comparable` interface, making it suitable
 * for use in collections or algorithms requiring sorted data.
 *
 * @constructor Creates a `ComparableNDArray` instance with the specified `ndarray`.
 * @property ndarray The underlying `NDArray` object encapsulated by this class.
 */
@JvmInline
value class ComparableNDArray(val ndarray: NDArray<Int, *>) : Comparable< NDArray<Int, *>> {
    override fun compareTo(other:  NDArray<Int, *>): Int {
        val a = ndarray.data
        val b = other.data

        if (a.size != b.size) return a.size.compareTo(b.size)

        for (i in a.indices) {
            val cmp = a[i].compareTo(b[i])
            if (cmp != 0) return cmp
        }
        return 0
    }

    fun toNDArray(): NDArray<Int, *> = ndarray

    override fun toString() = ndarray.toString()
}

/**
 * Converts the current object to a `Comparable` instance.
 *
 * If the object implements `Comparable`, it is returned as is.
 * If the object is an `NDArray`, it is wrapped in a `ComparableNDArray`.
 * For other types, an error is thrown indicating that the object
 * must either be `Comparable` or mappable to a comparable key.
 *
 * @return This object as a `Comparable` instance, or a wrapped `Comparable` in the case of an `NDArray`.
 * @throws IllegalStateException if the object is neither `Comparable` nor mappable to a comparable key.
 */
@Suppress("UNCHECKED_CAST")
internal fun <T> T.toComparable(): Comparable<T> =
    when (this) {
        is NDArray<*, *> -> ComparableNDArray(this as NDArray<Int, *>)
        is Comparable<*> -> this
        else -> error("Value $this must be Comparable or mappable to a comparable key.")
    } as Comparable<T>

@Suppress("UNCHECKED_CAST")
internal fun <T> Comparable<T>.fromComparable(): T =
    when (this) {
        is ComparableNDArray -> this.toNDArray()
        else -> this
    }  as T
