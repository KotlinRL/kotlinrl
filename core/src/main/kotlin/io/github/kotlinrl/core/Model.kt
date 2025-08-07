package io.github.kotlinrl.core

/**
 * Type alias for `io.github.kotlinrl.core.model.EmpiricalMDPModel`.
 *
 * Represents an empirical approach to modeling Markov Decision Processes (MDPs),
 * using sampling-based strategies to approximate transition dynamics and expected rewards.
 *
 * @param State The type representing states in the MDP.
 * @param Action The type representing actions in the MDP.
 */
typealias EmpiricalMDPModel<State, Action> = io.github.kotlinrl.core.model.EmpiricalMDPModel<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.model.LearnableMDPModel`.
 *
 * Represents a learnable Markov Decision Process (MDP) model that extends the capabilities
 * of a basic MDP model by incorporating functionality for learning state transition dynamics.
 *
 * This alias simplifies references to the `LearnableMDPModel` interface, which allows
 * for updating transition knowledge, sampling transitions, and determining if specific
 * state-action pairs are known within the model, facilitating the development of
 * reinforcement learning algorithms and environments.
 *
 * @param State The type representing the states of the environment modeled by the MDP.
 * @param Action The type representing the actions available in the MDP.
 */
typealias LearnableMDPModel<State, Action> = io.github.kotlinrl.core.model.LearnableMDPModel<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.model.MDPModel`.
 *
 * Represents a Markov Decision Process (MDP) model which defines the states, actions,
 * transitions, and rewards within the MDP. This alias provides a simplified reference
 * to the `MDPModel` interface in the `io.github.kotlinrl.core.model` package.
 *
 * @param State The type representing the states in the MDP.
 * @param Action The type representing the actions in the MDP.
 */
typealias MDPModel<State, Action> = io.github.kotlinrl.core.model.MDPModel<State, Action>
/**
 * Provides a type alias for the `ProbabilisticTransition` data class from the
 * `io.github.kotlinrl.core.model` package.
 *
 * This alias represents a probabilistic transition in a Markov Decision Process (MDP),
 * encapsulating the state, action, resulting state, reward, transition probability,
 * and whether the state is terminal. It simplifies access to the `ProbabilisticTransition`
 * data class, which is commonly used to model transitions in reinforcement learning
 * environments.
 *
 * @param State The type representing the states in the MDP.
 * @param Action The type representing the actions in the MDP.
 */
typealias ProbabilisticTransition<State, Action> = io.github.kotlinrl.core.model.ProbabilisticTransition<State, Action>
/**
 * Represents a probabilistic trajectory through a sequence of states and actions in an environment.
 *
 * This type alias defines a trajectory as a list of `ProbabilisticTransition` instances, which encode
 * transitions between states with associated actions and their probabilities. It is commonly used
 * in reinforcement learning or environments with stochastic dynamics to capture the behavior and
 * decision-making of an agent.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions in the environment.
 */
typealias ProbabilisticTrajectory<State, Action> = List<ProbabilisticTransition<State, Action>>
