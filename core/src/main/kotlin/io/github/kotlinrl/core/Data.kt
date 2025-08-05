package io.github.kotlinrl.core

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
