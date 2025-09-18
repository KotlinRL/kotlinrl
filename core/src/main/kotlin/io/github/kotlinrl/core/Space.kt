package io.github.kotlinrl.core

/**
 * Typealias representing a multi-dimensional, bounded space for sampling and validation
 * of numeric states and actions. It serves as a shorthand for the `Box` class in the
 * `io.github.kotlinrl.core.space` package.
 *
 * @param State The type of the state values in the space, constrained to subclasses of `Number`.
 * @param Action The type of the action values in the space.
 */
typealias Box<State, Action> = io.github.kotlinrl.core.space.Box<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.space.Dict`.
 *
 * Represents a dictionary-like space where each key corresponds to a subspace,
 * allowing composite representations of spaces in reinforcement learning or
 * similar frameworks. This alias simplifies access to the `Dict` class, which
 * aggregates multiple `Space` instances under a map-like structure, supporting
 * operations like sampling and containment checks across all subspaces.
 */
typealias Dict = io.github.kotlinrl.core.space.Dict
/**
 * Type alias for the `Discrete` class.
 *
 * Represents a discrete space of integers, characterized by a finite number of actions or entities.
 * It allows defining a range of integers and provides the capability for value sampling and
 * containment checks. The primary purpose of this alias is to simplify usage and improve
 * readability in code by referencing the fully qualified name from the `kotlinrl` library.
 */
typealias Discrete = io.github.kotlinrl.core.space.Discrete
/**
 * Represents a type alias for the `Graph` class in the `io.github.kotlinrl.core.space` package.
 *
 * @param N The type of features associated with the nodes in the graph.
 * @param E The type of features associated with the edges in the graph.
 */
typealias Graph<N, E> = io.github.kotlinrl.core.space.Graph<N, E>
/**
 * A type alias for the `io.github.kotlinrl.core.space.MultiBinary` class.
 *
 * Represents a binary-valued space with a fixed number of binary elements `n`.
 * Each element in the space independently takes a value of 0 or 1. This alias provides
 * a convenient reference to the `MultiBinary` class, commonly used in contexts such as
 * reinforcement learning to define action or observation spaces.
 */
typealias MultiBinary = io.github.kotlinrl.core.space.MultiBinary
/**
 * Type alias for `io.github.kotlinrl.core.space.MultiDiscrete`, representing a multi-dimensional discrete space.
 *
 * `MultiDiscrete` is a space consisting of multiple discrete dimensions, where each dimension has a finite range
 * of integers defined by the `nvec` array. It provides functionality for sampling random values within these ranges
 * and checking the validity of values against the space's definition.
 */
typealias MultiDiscrete = io.github.kotlinrl.core.space.MultiDiscrete
/**
 * Type alias for the `OneOf` class from the KotlinRL library.
 *
 * Represents a composite space consisting of multiple subspaces. Samples
 * are drawn randomly from one of the subspaces based on a uniform
 * probability distribution. Provides functionality for sampling values
 * and verifying membership within the composite space.
 */
typealias OneOf = io.github.kotlinrl.core.space.OneOf
/**
 * A type alias for `io.github.kotlinrl.core.space.Sequence<T>`.
 *
 * Represents a sequence space that can generate randomly sampled sequences
 * of a bounded maximum length from a given space. Additionally, provides a mechanism
 * for checking if a value belongs to the sequence space.
 *
 * @param T The type of elements in the sequence.
 */
typealias Sequence<T> = io.github.kotlinrl.core.space.Sequence<T>
/**
 * Type alias for `io.github.kotlinrl.core.space.Space`.
 *
 * Represents a generic space of type `T`, providing methods for sampling elements
 * and checking the membership of elements within the space. This alias simplifies
 * the reference to the `Space` interface for easier usage within the codebase.
 *
 * @param T The type of elements the space contains.
 */
typealias Space<T> = io.github.kotlinrl.core.space.Space<T>
/**
 * Type alias for the `io.github.kotlinrl.core.space.Text` class.
 *
 * Represents a space of strings, where strings are constrained by a maximum length and
 * a defined set of allowable characters. This type alias provides a shorthand for referencing
 * the `Text` class in contexts where text space manipulation or evaluation is required.
 */
typealias Text = io.github.kotlinrl.core.space.Text
/**
 * Type alias for the `Tuple` class, which represents a composite space consisting
 * of multiple individual `Space` instances. It is used to manage and validate
 * a list of elements where each corresponds to one of the contained spaces.
 */
typealias Tuple = io.github.kotlinrl.core.space.Tuple