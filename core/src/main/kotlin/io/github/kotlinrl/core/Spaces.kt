package io.github.kotlinrl.core

/**
 * Type alias for the `Box` class. This alias is a shorthand to reference the `Box` class
 * from `io.github.kotlinrl.core.space` package.
 *
 * The `Box` class represents a bounded space defined by lower and upper bounds
 * for each dimension. It allows for sampling values within the defined bounds
 * and verifying if a given value lies within the space.
 *
 * @param T The numeric type of the values (e.g., Int, Float, Double, etc.), constrained to subclasses of `Number`.
 * @param D The dimensionality of the space, constrained to `Dimension`.
 */
typealias Box<T, D> = io.github.kotlinrl.core.space.Box<T, D>
/**
 * Type alias for `io.github.kotlinrl.core.space.Dict`, representing a dictionary-like space
 * where each key is mapped to a specific subspace. This alias provides a concise reference
 * for the class, which aggregates multiple subspaces defined by the `Space` interface.
 *
 * `Dict` is commonly used to define composite spaces with independent subspace sampling
 * and validation logic, offering flexibility in specifying complex structured spaces.
 */
typealias Dict = io.github.kotlinrl.core.space.Dict
/**
 * Typealias for `io.github.kotlinrl.core.space.Discrete`, providing a simplified name for
 * accessing the `Discrete` class within the codebase.
 *
 * The `Discrete` class represents a discrete space with a finite number of integers,
 * starting from a specified value. It supports sampling random integers within the defined range
 * and checking if a certain value is included in the space. This typealias improves readability
 * and provides a shorthand reference to the `Discrete` class.
 */
typealias Discrete = io.github.kotlinrl.core.space.Discrete
/**
 * Type alias for the `io.github.kotlinrl.core.space.Graph` class.
 *
 * This alias provides a shorthand representation for the graph space which models
 * graphs containing nodes and edges with associated features.
 *
 * @param N The type of features associated with nodes in the graph.
 * @param E The type of features associated with edges in the graph.
 */
typealias Graph<N, E> = io.github.kotlinrl.core.space.Graph<N, E>
/**
 * Type alias for `io.github.kotlinrl.core.space.MultiBinary`, representing a binary-valued space
 * with a fixed size `n`. Each element in the space can independently take the value 0 or 1.
 *
 * This alias provides a streamlined reference to the `MultiBinary` class, which includes
 * functionality for sampling random binary values and validating whether a value belongs
 * to the defined space.
 */
typealias MultiBinary = io.github.kotlinrl.core.space.MultiBinary
/**
 * Typealias for `io.github.kotlinrl.core.space.MultiDiscrete`, which represents a space with multiple
 * discrete dimensions. Each dimension is defined by a finite range of integers specified by its size.
 *
 * This alias simplifies the reference to `MultiDiscrete` within the codebase, encapsulating
 * the concept of multi-dimensional discrete spaces commonly used in reinforcement learning
 * or simulation contexts.
 */
typealias MultiDiscrete = io.github.kotlinrl.core.space.MultiDiscrete
/**
 * Type alias for `io.github.kotlinrl.core.space.OneOf`, representing a composite space
 * consisting of multiple subspaces. Samples are drawn randomly from one of the subspaces
 * using a uniform probability distribution.
 *
 * This alias simplifies references to the `OneOf` class and is primarily used
 * to define or work with spaces that involve sampling from a mixture of subspaces. Each
 * subspace can define its own structure or type, allowing for flexible and diverse
 * configurations of the sampling space.
 */
typealias OneOf = io.github.kotlinrl.core.space.OneOf
/**
 * Typealias for `io.github.kotlinrl.core.space.Sequence`.
 *
 * Represents a sequence space capable of generating random sequences of elements sampled
 * from a defined space with a bounded maximum length. Provides methods to generate
 * random samples from the space and validate if a sequence belongs to it.
 *
 * This alias simplifies references to the `Sequence` class, enabling concise usage
 * within the codebase.
 *
 * @param T The type of elements contained in the sequence space.
 */
typealias Sequence<T> = io.github.kotlinrl.core.space.Sequence<T>
/**
 * Type alias for `io.github.kotlinrl.core.space.Text`, representing a space of text strings with
 * constraints such as maximum length, character set, and optional random seed. This type alias
 * simplifies references to the `Text` class within the codebase.
 */
typealias Text = io.github.kotlinrl.core.space.Text
/**
 * Represents a type alias for `io.github.kotlinrl.core.space.Tuple`.
 *
 * This type alias simplifies referencing the `Tuple` class, which is a composite
 * space consisting of a sequence of individual spaces. Each space manages a specific
 * element type, enabling the `Tuple` to collectively handle a list of elements
 * belonging to different spaces.
 *
 * It provides functionalities to sample elements from all component spaces
 * and to verify membership of a list of values against the composite space.
 */
typealias Tuple = io.github.kotlinrl.core.space.Tuple
/**
 * A type alias for the `io.github.kotlinrl.core.space.Space` interface.
 *
 * Represents a generic space of type `T`, which provides functionalities
 * for sampling elements and verifying if a value belongs to the space.
 *
 * This alias simplifies the reference to the `Space` interface within the library or application.
 *
 * @param T The type of elements the space contains.
 */
typealias Space<T> = io.github.kotlinrl.core.space.Space<T>