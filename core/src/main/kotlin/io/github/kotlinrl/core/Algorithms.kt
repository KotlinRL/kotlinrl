package io.github.kotlinrl.core

/**
 * A type alias for the `LearningAlgorithm` interface defined in the `io.github.kotlinrl.core.algorithms.base` package.
 *
 * This alias simplifies references to the `LearningAlgorithm` interface, which is a core abstraction
 * representing reinforcement learning algorithms. It encompasses the operations required for learning
 * from environment interactions, including determining actions, processing transitions, and trajectory updates.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken in the environment.
 */
typealias LearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithm.LearningAlgorithm<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.algorithms.base.TransitionLearningAlgorithm`.
 *
 * Represents a reinforcement learning algorithm that updates the Q-function and policy
 * based on state-action transitions. This alias simplifies references to the core
 * `TransitionLearningAlgorithm` class, which provides functionality for incremental
 * learning methods that process individual transitions rather than complete trajectories.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be performed within the environment.
 */
typealias TransitionLearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithm.TransitionLearningAlgorithm<State, Action>
/**
 * Typealias for `io.github.kotlinrl.core.algorithms.base.TrajectoryLearningAlgorithm`.
 *
 * Represents a reinforcement learning algorithm that focuses on trajectory-based learning,
 * where updates to the policy and Q-function are performed using sequences of state-action-reward
 * transitions (trajectories). The algorithm works with on-policy updates, delegating Q-function
 * estimation to a trajectory-informed estimation process.
 *
 * Useful for scenarios where entire episodes or trajectories are leveraged for learning,
 * improving the decision-making policy over time based on observed data.
 *
 * @param State The type representing the environment states.
 * @param Action The type representing possible actions performed in the environment.
 */
typealias TrajectoryLearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithm.TrajectoryLearningAlgorithm<State, Action>
