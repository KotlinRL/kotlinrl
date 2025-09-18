package io.github.kotlinrl.core

/**
 * A type alias for `FiniteStates`, representing a finite set of integer states
 * commonly used in reinforcement learning or systems requiring state enumeration.
 *
 * Simplifies referencing the `FiniteStates` class with a shorter, contextual name.
 */
typealias FiniteState = io.github.kotlinrl.core.model.FiniteStates
/**
 * A type alias for the `io.github.kotlinrl.core.model.FiniteTabular` interface.
 *
 * This alias simplifies the usage of the `FiniteTabular` interface by providing an alternative
 * name within the current context. The `FiniteTabular` interface is designed to represent a
 * finite tabular Markov Decision Process (MDP) with a finite set of states and actions, where
 * both states and actions are integer-encoded and enumerable.
 *
 * It provides the foundation for working with structured MDPs, supporting the definition of
 * state and action spaces and facilitating interactions in reinforcement learning or decision-making
 * tasks. The alias references the `FiniteTabular` interface from its original declaration.
 */
typealias FiniteTabular = io.github.kotlinrl.core.model.FiniteTabular
/**
 * A type alias for `io.github.kotlinrl.core.model.FixedIntActions`.
 *
 * Represents a fixed set of integer actions available for a given state in a
 * reinforcement learning environment or related contexts. The number of actions
 * is predetermined, and this alias simplifies the reference to the corresponding
 * `FixedIntActions` class.
 */
typealias FixedIntActions = io.github.kotlinrl.core.model.FixedIntActions
/**
 * Type alias for `io.github.kotlinrl.core.model.MDP`.
 *
 * Represents a Markov Decision Process (MDP), a mathematical framework widely used
 * in reinforcement learning and decision-making under uncertainty. It models an
 * environment where an agent interacts by taking actions, transitioned through states,
 * and receiving rewards based on a defined reward function and transition probabilities.
 *
 * The alias simplifies access to the `MDP` interface and its associated state, action,
 * reward, transition, and discount factor properties used in reinforcement learning contexts.
 *
 * @param State The type representing the state space of the MDP.
 * @param Action The type representing the action space of the MDP.
 */
typealias MDP<State, Action> = io.github.kotlinrl.core.model.MDP<State, Action>
/**
 * A type alias representing a tabular Markov Decision Process (MDP).
 *
 * This alias simplifies the reference to the `TabularMDP` class, a core component in reinforcement
 * learning that models decision-making problems using defined states, actions, rewards, transitions,
 * and a discount factor. It is especially useful in tabular RL settings where finite state and
 * action spaces are assumed.
 */
typealias TabularMDP = io.github.kotlinrl.core.model.TabularMDP