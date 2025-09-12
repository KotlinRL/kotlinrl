package io.github.kotlinrl.core.api

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

