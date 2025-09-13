package io.github.kotlinrl.core

/**
 * Type alias for `io.github.kotlinrl.core.api.Actions`, representing a functional interface
 * for associating actions with a specific state.
 *
 * This alias simplifies the reference to the `Actions` functional interface, which is often
 * utilized in contexts such as decision-making or reinforcement learning to dynamically determine
 * the available actions for a given state.
 *
 * @param State The type representing the state for which actions are defined.
 * @param Action The type representing the actions associated with the state.
 */
typealias Actions<State, Action> = io.github.kotlinrl.core.api.Actions<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.api.Parameter`.
 *
 * Represents a parameter that tracks values such as the current state,
 * previous state, and minimum allowable value. Typically utilized
 * in iterative processes like parameter scheduling, optimization
 * algorithms, or reinforcement learning scenarios. Provides an alias
 * to simplify references to the `Parameter` class within the codebase.
 */
typealias Parameter = io.github.kotlinrl.core.api.Parameter
/**
 * Typealias for `io.github.kotlinrl.core.api.Policy`, representing a decision-making policy
 * in reinforcement learning or similar frameworks.
 *
 * The `Policy` maps a given state to an action or a probability distribution over possible actions,
 * offering an interface for deterministic or probabilistic action selection.
 *
 * @param State The type representing the states over which the policy operates.
 * @param Action The type representing the actions determined by the policy.
 */
typealias Policy<State, Action> = io.github.kotlinrl.core.api.Policy<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.api.States`.
 *
 * Represents a collection of states in a reinforcement learning environment or similar systems.
 * This alias is primarily used to provide a more concise reference to the `States` interface,
 * which includes properties and methods for working with finite or infinite state spaces,
 * checking state membership, and iterating over states.
 *
 * @param State The type representing the states in the environment.
 */
typealias States<State> = io.github.kotlinrl.core.api.States<State>
/**
 * A type alias representing a value function in reinforcement learning.
 *
 * This function maps a given state to a `Double` value, typically representing
 * the expected cumulative reward or value associated with that state.
 *
 * @param State The type representing the state space of the environment.
 */
typealias V<State> = (State) -> Double
/**
 * Represents a value function for a given state in reinforcement learning or decision-making frameworks.
 *
 * A state-value function maps each state to a value, typically representing the expected total reward
 * (or utility) of being in that state under a certain policy or scenario. This type alias provides
 * a shorthand for defining functions that calculate or store state values.
 *
 * @param State The type representing the state space.
 */
typealias StateValueFunction<State> = V<State>
/**
 * A type alias representing a state-action value function in reinforcement learning.
 *
 * This function maps a given state and action pair to a `Double` value, typically representing
 * the expected cumulative reward or value associated with taking that action in the given state.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias Q<State, Action> = (State, Action) -> Double
/**
 * Represents a function that calculates the value of an action in a given state.
 *
 * This type alias can be used in reinforcement learning applications to define
 * or work with value functions, such as Q-functions, which estimate the value
 * of taking a particular action in a specific state.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be taken in the environment.
 */
typealias ActionValueFunction<State, Action> = Q<State, Action>
/**
 * A type alias representing a reward function in reinforcement learning.
 *
 * This function maps a given state-action pair to a `Double` value, which typically represents
 * the immediate reward obtained for taking the specified action in the given state.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias R<State, Action> = (State, Action) -> Double
/**
 * Represents a reward function in the context of reinforcement learning or similar frameworks.
 *
 * This type alias defines a function that evaluates the reward for a given state-action pair,
 * enabling dynamic computation of rewards during the learning or decision-making process.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions in the environment.
 */
typealias RewardFunction<State, Action> = R<State, Action>
/**
 * A type alias representing a state-action-state transition probability function in reinforcement learning.
 *
 * This function defines the likelihood of transitioning from a given state to another state
 * after taking a specific action. Typically used in the definition of Markov Decision Processes (MDPs),
 * where the transition function provides the probability distribution over next states.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @return A `Double` value representing the transition probability for the given state-action-state triplet.
 */
typealias T<State, Action> = (State, Action, State) -> Double
/**
 * Represents a type alias for a function defining transition dynamics in a reinforcement
 * learning context or similar systems.
 *
 * This alias simplifies the representation of a function that maps state-action pairs
 * to a probabilistic outcome or a resulting state distribution, which describes
 * the dynamics of the environment or system under consideration.
 *
 * @param State The type representing the states within the environment.
 * @param Action The type representing the actions taken within the environment.
 */
typealias TransitionDynamicsFunction<State, Action> = T<State, Action>
/**
 * A type alias representing a callback function to be notified when a given policy is updated.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing the actions that can be taken within the environment.
 */
typealias PolicyUpdate<State, Action> = (Policy<State, Action>) -> Unit
/**
 * A type alias representing a decay function for a parameter schedule.
 *
 * This function updates the internal state of the schedule to reflect
 * a change corresponding to one step or iteration. Typically used in
 * reinforcement learning algorithms for dynamically adjusting parameters
 * such as exploration rates or learning rates over time.
 */
typealias ParameterScheduleDecay = () -> Unit

