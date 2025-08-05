package io.github.kotlinrl.core

import kotlin.random.*

/**
 * A type alias representing a reinforcement learning algorithm that computes actions based on states
 * and updates its policy according to observed transitions and trajectories.
 *
 * This abstraction provides functionality for:
 * - Invoking the algorithm with a state to obtain an action.
 * - Updating the algorithm's policy based on individual transitions or trajectories.
 *
 * The actual implementation is defined in the `io.github.kotlinrl.core.algorithms.base.LearningAlgorithm` class.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be performed in the environment.
 */
typealias LearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.LearningAlgorithm<State, Action>
/**
 * A type alias for `QFunctionAlgorithm` class, which serves as the base class for algorithms
 * utilizing a Q-Function for decision-making in reinforcement learning. The Q-Function maps
 * state-action pairs to expected rewards, facilitating learning and optimization processes.
 *
 * This base class supports both on-policy and off-policy updates and is integrated with
 * a `QFunctionPolicy` for policy handling. The `Q` property of the algorithm maintains the
 * Q-values used during the learning process, and updates trigger a configurable handler for
 * processing Q-function changes in real-time.
 *
 * @param State The type parameter representing the state space handled by the algorithm.
 * @param Action The type parameter representing the action space handled by the algorithm.
 */
typealias QFunctionAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.QFunctionAlgorithm<State, Action>
/**
 * Type alias for `TDQError`, a functional interface used to compute Temporal Difference (TD) Q-errors
 * in reinforcement learning.
 *
 * The alias represents a function that takes various parameters such as the Q-function, a transition,
 * the next action, discount factor (`gamma`), and a flag indicating if the episode is done, and returns
 * a computed TD-error (as a `Double`).
 *
 * This is commonly used in TD-based algorithms like Q-learning or SARSA.
 */
typealias TDQError<State, Action> = io.github.kotlinrl.core.algorithms.td.TDQError<State, Action>
/**
 * A type alias for the `TDVError` functional interface within the Temporal Difference (TD) framework.
 *
 * `TDVError` is used to compute the Temporal Difference error in reinforcement learning, which
 * quantifies the difference between predicted and actual rewards under a Markov Decision Process (MDP).
 * It interprets the current value function, a state transition, and a discount factor to produce the TD error,
 * aiding in the evaluation and improvement of value function estimations.
 *
 * @param State The type representing the state space of the MDP.
 */
typealias TDVError<State> = io.github.kotlinrl.core.algorithms.td.TDVError<State>
/**
 * Alias for `TDQErrors` in the `io.github.kotlinrl.core.algorithms.td` package.
 *
 * Represents the type used for describing Temporal Difference (TD) error computations
 * in reinforcement learning algorithms. TD errors are used in updating Q-values
 * based on the difference between estimated and observed rewards.
 */
typealias TDQErrors = io.github.kotlinrl.core.algorithms.td.TDQErrors
/**
 * Type alias for `TDVErrors` in the context of Temporal Difference (TD) learning.
 * Represents data or object structures used to manage or compute errors while applying
 * TD-based reinforcement learning algorithms.
 */
typealias TDVErrors = io.github.kotlinrl.core.algorithms.td.TDVErrors
/**
 * A type alias for the `TransitionQFunctionAlgorithm` class.
 *
 * Represents an abstract class for reinforcement learning algorithms that utilize
 * Q-function estimations driven by observed state-action transitions. It combines
 * a Q-function estimation strategy with policy improvement to iteratively refine
 * both the policy and the Q-function.
 *
 * This alias is used to simplify references to the implementation in the codebase.
 *
 * Generic parameters:
 * - `State`: The type representing an environment's state.
 * - `Action`: The type representing an action within the environment.
 */
typealias TransitionQFunctionAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.TransitionQFunctionAlgorithm<State, Action>
/**
 * A type alias for `TransitionQFunctionEstimator` interface, which provides a method
 * to estimate a new `EnumerableQFunction` based on a given Q-function and a state-action transition.
 *
 * This interface is commonly utilized in reinforcement learning algorithms where
 * Q-functions are updated based on observed transitions within the environment.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the action to be taken in the environment.
 */
typealias TransitionQFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TransitionQFunctionEstimator<State, Action>
/**
 * Represents a type alias for `TrajectoryQFunctionAlgorithm` class.
 * This abstraction defines a Q-learning based trajectory prediction algorithm,
 * designed for reinforcement learning. It utilizes a policy and a trajectory-based
 * Q-function estimator to update both the Q-function and policy iteratively.
 *
 * The class implements the observation of trajectories and uses the trajectory
 * data to refine the Q-function and improve decision-making policies.
 *
 * This type alias simplifies the reference to the fully qualified class name
 * within the codebase.
 *
 * @param State The type representing the state in the environment.
 * @param Action The type representing the action in the environment.
 */
typealias TrajectoryQFunctionAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionAlgorithm<State, Action>
/**
 * Type alias for the `TrajectoryQFunctionEstimator` interface, which provides functionality
 * to estimate Q-values for given state-action pairs based on a trajectory.
 *
 * The interface is used to estimate an enumerable Q-function (`EnumerableQFunction`) using
 * a specified trajectory (`Trajectory`), enabling reinforcement learning algorithms
 * to update their Q-value representations based on observed experiences.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions in the environment.
 */
typealias TrajectoryQFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionEstimator<State, Action>
/**
 * Type alias for the `TrajectoryValueFunctionEstimator` interface.
 *
 * `TrajectoryValueFunctionEstimator` represents a contract for estimating the value function
 * from a provided trajectory in a reinforcement learning context. The implementation is expected
 * to update or compute an enumerable value function (`V`) based on observed state-action trajectories.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias TrajectoryValueFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TrajectoryValueFunctionEstimator<State, Action>
/**
 * A type alias for the `TransitionValueFunctionEstimator` interface.
 *
 * This interface defines a mechanism for estimating updated state value functions
 * given an enumerable value function and a state-action transition.
 *
 * It provides abstraction for algorithms that require estimation of
 * value function updates influenced by transitions in a reinforcement learning context.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions in the environment.
 *
 * Features:
 * - Accepts an input value function (`V`) which maps states to values.
 * - Considers a given transition consisting of state-action pairs.
 * - Outputs an updated value function reflecting changes induced by the transition.
 */
typealias TransitionValueFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TransitionValueFunctionEstimator<State, Action>
/**
 * Type alias for `BellmanValueFunctionIteration`, a Dynamic Programming algorithm used in
 * Reinforcement Learning for solving Markov Decision Processes (MDPs). It iteratively updates
 * the value function for all states in the MDP using the Bellman equation until convergence.
 *
 * This algorithm works by calculating the expected return of taking the best action possible
 * for each state-action pair, based on the current value function and model transitions, ensuring
 * that the policy derived is optimal for the given MDP.
 *
 * State: The type representing the states in the MDP.
 * Action: The type representing the actions in the MDP.
 */
typealias BellmanValueFunctionIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanValueFunctionIteration<State, Action>
/**
 * Type alias for `BellmanQFunctionIteration`, which is a dynamic programming approach
 * for iteratively updating and computing the Q-function in reinforcement learning.
 *
 * This type represents a specific implementation of Bellman updates, utilizing
 * an MDP model, state-action space, and a configurable discount factor (`gamma`)
 * and convergence threshold (`theta`). Commonly used for policy evaluation
 * or policy improvement operations.
 */
typealias BellmanQFunctionIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanQFunctionIteration<State, Action>
/**
 * A type alias for the `BellmanPolicyIteration` class, used in dynamic programming
 * for solving Markov Decision Processes (MDPs) through policy iteration.
 *
 * This method iteratively evaluates and improves a given policy based on the Bellman equation,
 * ensuring convergence to an optimal policy under specific conditions.
 *
 * @param State The type representing states in the MDP.
 * @param Action The type representing actions in the MDP.
 */
typealias BellmanPolicyIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanPolicyIteration<State, Action>
/**
 * Type alias for the `OnPolicyMonteCarloControl` class, which implements the On-Policy Monte Carlo
 * Control algorithm for reinforcement learning.
 *
 * This algorithm estimates the state-action value function (Q-function) using complete episodes
 * or trajectories by averaging returns (rewards) for each state-action pair.
 *
 * Key features include:
 * - Support for first-visit or every-visit Monte Carlo methods.
 * - Integration with a policy derived from a Q-function using an epsilon-greedy approach or similar.
 * - Modular design allowing customization of Q-function updates and policy updates.
 *
 * This alias provides a concise way to reference the full class within the codebase and is typed
 * for specific `State` and `Action` definitions.
 */
typealias OnPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OnPolicyMonteCarloControl<State, Action>
/**
 * Type alias for `IncrementalMonteCarloControl` in the `io.github.kotlinrl.core.algorithms.mc` package.
 *
 * Represents a reinforcement learning algorithm based on Incremental Monte Carlo control.
 * It is used for learning an optimal policy by iteratively improving the Q-function and
 * policy based on accumulated experience (trajectories). The algorithm supports incremental
 * updates using learning rate (`alpha`), discount factor (`gamma`), and can operate in
 * first-visit mode.
 *
 * This alias provides a shorthand reference to the full class, simplifying usage and improving code readability.
 */
typealias IncrementalMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.IncrementalMonteCarloControl<State, Action>
/**
 * Type alias for `OffPolicyMonteCarloControl` class, part of the reinforcement learning algorithm suite.
 * This class implements an off-policy Monte Carlo control algorithm for learning Q-values
 * based on sample trajectories from a behavioral policy while improving a target policy.
 *
 * The algorithm supports:
 * - A behavioral policy for generating trajectories.
 * - A target policy, which is improved iteratively.
 * - Discount factor `gamma` to account for future rewards.
 * - Configurable trajectory Q-function estimators and callbacks for updates.
 *
 * The off-policy approach utilizes importance sampling techniques to correct for
 * the mismatch between the behavioral policy and the target policy.
 * It is suitable for problems requiring the learning agent to evaluate and improve policies
 * from exploratory data without fully relying on following the target policy.
 */
typealias OffPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OffPolicyMonteCarloControl<State, Action>
/**
 * Type alias for the ExpectedSARSA class in the Classic Temporal Difference (TD) learning algorithms package.
 *
 * Expected SARSA is a model-free reinforcement learning algorithm used for estimating the action-value
 * function (Q-function) in environments where exact dynamics are not known. This alias provides
 * a convenient shorthand for the fully-qualified class reference within the codebase.
 *
 * The ExpectedSARSA algorithm combines elements of TD learning and the policy evaluation
 * phase of SARSA with an expectation over actions, rather than relying on a sampled next action.
 * It uses this expectation to compute the TD target, making it more stable and potentially
 * leading to better convergence properties compared to SARSA.
 *
 * This alias assumes the use of the following components:
 * - `initialPolicy`: The initial policy, represented as a Q-function policy.
 * - `alpha`: A parameter schedule controlling the learning rate.
 * - `gamma`: The discount factor for future rewards.
 * - `estimator`: A transition Q-function estimator, defaulting to the ExpectedSARSAQFunctionEstimator.
 * - `onQFunctionUpdate` and `onPolicyUpdate`: Callbacks for Q-function and policy updates, respectively.
 *
 * The ExpectedSARSA algorithm operates on states (`State`) and actions (`Action`) as generic parameters,
 * enabling flexibility in application to various tasks and environments.
 */
typealias ExpectedSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.ExpectedSARSA<State, Action>
/**
 * A type alias for the `QLearning` class, representing the Q-Learning algorithm
 * implementation for reinforcement learning. Q-Learning is an off-policy, temporal-difference
 * learning algorithm used to learn the optimal policy for an agent interacting with an environment.
 *
 * This alias provides a more concise and readable reference for the implementation of
 * Q-Learning within the codebase. The class supports customizable policies, learning rate schedules,
 * discount factors, Q-function estimators, and hooks for policy and Q-function updates.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the agent.
 */
typealias QLearning<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.QLearning<State, Action>
/**
 * Type alias for the `SARSA` class, a reinforcement learning algorithm that performs state-action
 * value updates based on the SARSA (State-Action-Reward-State-Action) update rule. It is a form of
 * Temporal Difference (TD) learning.
 *
 * The SARSA algorithm utilizes the current policy to update the Q-function and is often called an
 * on-policy learning algorithm. It incrementally learns the Q-value function for a given
 * environment while interacting with it.
 *
 * This alias simplifies the reference to the `SARSA` implementation within the codebase.
 *
 * Parameters of the underlying SARSA class include:
 * - `initialPolicy`: Initial policy for selecting actions in each state.
 * - `alpha`: Step size parameter or learning rate.
 * - `gamma`: Discount factor dictating the consideration of future rewards.
 * - `estimator`: The transition Q-function estimator, used to compute updates.
 * - `onQFunctionUpdate`: Callback invoked after updating the Q-function.
 * - `onPolicyUpdate`: Callback invoked after updating the policy.
 */
typealias SARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.SARSA<State, Action>
/**
 * Type alias for `NStepSARSA`, a reinforcement learning algorithm implementing the
 * n-step SARSA (State-Action-Reward-State-Action) method within the context of Temporal Difference (TD) learning.
 *
 * `NStepSARSA` is a temporal difference learning strategy that updates the Q-function using n-step transitions,
 * balancing between Monte Carlo and bootstrapping techniques. It leverages a specific policy, learning rate (alpha),
 * discount factor (gamma), and a trajectory-based Q-value estimator.
 *
 * The algorithm supports custom update handling for both the Q-function and policy, allowing flexible integration
 * into different reinforcement learning frameworks.
 */
typealias NStepSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.nstep.NStepSARSA<State, Action>
/**
 * Type alias for `DPIteration` class, which serves as a fundamental building block for dynamic programming
 * algorithms in reinforcement learning. The `DPIteration` class is an abstract implementation of a planner
 * that computes and returns a policy for a given environment. The planning process typically involves iterative
 * methods to optimize the policy based on the problem's dynamics and reward structure.
 *
 * This alias provides a concise reference to the class while maintaining type parameterization for state and action.
 *
 * @param State Represents the type used for states in the environment.
 * @param Action Represents the type used for actions in the environment.
 */
typealias DPIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.DPIteration<State, Action>
/**
 * A type alias for `DPValueFunctionEstimator`, which provides functionality to estimate
 * updated value functions in dynamic programming approaches for reinforcement learning.
 *
 * This estimator takes an existing value function and a probabilistic trajectory as inputs
 * and computes an adjusted value function.
 *
 * - `State`: The type representing the state space in the environment.
 * - `Action`: The type representing the action space in the environment.
 */
typealias DPValueFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.dp.DPValueFunctionEstimator<State, Action>
/**
 * Defines a type alias for a function responsible for updating an `EnumerableQFunction`.
 *
 * An `EnumerableQFunction` represents a Q-function that can enumerate all possible
 * state-action pairs. This alias encapsulates the pattern where updates or modifications
 * can be made to an existing `EnumerableQFunction` instance through a custom implementation.
 *
 * @param State Represents the state space of the Q-function.
 * @param Action Represents the action space of the Q-function.
 */
typealias EnumerableQFunctionUpdate<State, Action> = (EnumerableQFunction<State, Action>) -> Unit
/**
 * Represents a function type that enables updates to an `EnumerableValueFunction` of a given state type.
 *
 * This type alias is used to define a functional transformation or modification that can be applied to
 * an `EnumerableValueFunction`, which is a function mapping enumerable states to values.
 *
 * @param State The type of the state associated with the `EnumerableValueFunction`.
 */
typealias EnumerableValueFunctionUpdate<State> = (EnumerableValueFunction<State>) -> Unit


/**
 * Performs Bellman value function iteration for a given Markov Decision Process (MDP) environment.
 * This method iteratively computes the optimal state-value function using Bellman's equation.
 *
 * @param State The type representing the states of the MDP.
 * @param Action The type representing the actions of the MDP.
 * @param initialV The initial value function over the enumerable state space.
 * @param env The model-based environment used to simulate state transitions and rewards.
 * @param numSamples The number of samples used to approximate the transition dynamics and rewards. Defaults to 100.
 * @param gamma The discount factor for future rewards, defining the importance of future rewards relative to immediate rewards. Defaults to 0.99.
 * @param theta The convergence threshold for the value function iteration. Iteration terminates when the maximum change in value function across states is less than this threshold
 * . Defaults to 1e-6.
 * @param stateActions A mapping from states to the set of possible actions in each state.
 * @param onValueFunctionUpdate A callback triggered after each value function update during the iteration process. Defaults to no-op.
 * @return A `DPIteration` object representing the constructed result of the dynamic programming iteration process, including the policy and value function.
 */
fun <State, Action> bellmanValueFunctionIteration(
    initialV: EnumerableValueFunction<State>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
): DPIteration<State, Action> = BellmanValueFunctionIteration(
    initialV = initialV,
    model = EmpiricalMDPModel(
        env = env,
        allStates = initialV.allStates(),
        allActions = initialV.allStates().flatMap { stateActions(it) }.toList(),
        numSamples = numSamples
    ),
    gamma = gamma,
    theta = theta,
    stateActions = stateActions,
    onValueFunctionUpdate = onValueFunctionUpdate
)

/**
 * Performs the Bellman Q-function iteration algorithm to find an optimal policy
 * for a given model-based environment based on the Q-learning approach.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions in the environment.
 * @param initialQ An instance of `EnumerableQFunction` representing the initial Q-function
 *                 values for state-action pairs.
 * @param env The model-based environment for simulation and evaluation, used to
 *            approximate transition dynamics and rewards.
 * @param numSamples The number of samples to use for approximating transitions and rewards in
 *                   the empirical model. Defaults to 100.
 * @param gamma The discount factor for future rewards, commonly between 0 and 1. Defaults to 0.99.
 * @param theta A convergence threshold for stopping the iteration. The iteration halts
 *              if the maximum change between Q-values of successive iterations is less than this value.
 *              Defaults to 1e-6.
 * @param stateActions A function that provides the list of all possible actions for a given
 *                     state in the environment.
 * @param onQFunctionUpdate A callback executed after every Q-function update during the iteration.
 *                          Defaults to an empty lambda.
 *
 * @return An instance of `DPIteration` that contains the results of the Bellman Q-function iteration
 *         process and the derived optimal policy.
 */
fun <State, Action> bellmanQFunctionIteration(
    initialQ: EnumerableQFunction<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
): DPIteration<State, Action> = BellmanQFunctionIteration(
    initialQ = initialQ,
    model = EmpiricalMDPModel(
        env = env,
        allStates = initialQ.allStates(),
        allActions = initialQ.allStates().flatMap { stateActions(it) }.toList(),
        numSamples = numSamples
    ),
    gamma = gamma,
    theta = theta,
    stateActions = stateActions,
    onQFunctionUpdate = onQFunctionUpdate
)

/**
 * Performs policy iteration using Bellman's approach to improve a policy and its value function
 * for a given model-based environment. This algorithm iteratively evaluates a policy and updates
 * it until a stable (optimal) policy is found.
 *
 * @param initialV The initial value function as an enumerable value function over the state space.
 * @param initialPolicy The initial policy to be improved during the policy iteration process.
 * @param env The model-based environment representing the problem dynamics.
 * @param numSamples The number of samples to use for approximating transitions and rewards. Default is 100.
 * @param gamma The discount factor for future rewards, controlling how much future rewards are valued. Default is 0.99.
 * @param theta The convergence threshold for policy evaluation. Smaller values lead to higher precision but more iterations. Default is 1e-6.
 * @param stateActions A function that maps a state to the list of possible actions in that state.
 * @param onValueFunctionUpdate A callback invoked upon value function updates during policy evaluation. Default is an empty lambda.
 * @param onPolicyUpdate A callback invoked upon policy updates during improvement. Default is an empty lambda.
 * @return The resulting policy iteration algorithm interface that, when executed, finds the optimal policy for the given environment.
 */
fun <State, Action> bellmanPolicyIteration(
    initialV: EnumerableValueFunction<State>,
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): DPIteration<State, Action> = BellmanPolicyIteration(
    initialPolicy = initialPolicy,
    initialV = initialV,
    model = EmpiricalMDPModel(
        env = env,
        allStates = initialV.allStates(),
        allActions = initialV.allStates().flatMap { stateActions(it) }.toList(),
        numSamples = numSamples
    ),
    gamma = gamma,
    theta = theta,
    stateActions = stateActions,
    onValueFunctionUpdate = onValueFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)

/**
 * Implements the on-policy Monte Carlo control method for reinforcement learning.
 * This method iteratively improves a policy based on observed trajectories and
 * updates the associated Q-function.
 *
 * @param initialPolicy the initial policy to be used for decision-making and updated during learning
 * @param gamma the discount factor applied to future rewards, where 0 ≤ gamma ≤ 1
 * @param firstVisitOnly if true, only the first occurrence of a state-action pair in a trajectory
 *        will be used for updates; if false, all occurrences will be used
 * @param onQFunctionUpdate an optional callback invoked upon each update of the Q-function,
 *        providing the updated values
 * @param onPolicyUpdate an optional callback invoked upon each update of the policy,
 *        providing the new policy state
 * @return a configured [TrajectoryQFunctionAlgorithm] instance that performs the on-policy
 *         Monte Carlo control method
 */
fun <State, Action> onPolicyMonteCarloControl(
    initialPolicy: QFunctionPolicy<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): TrajectoryQFunctionAlgorithm<State, Action> = OnPolicyMonteCarloControl(
    initialPolicy = initialPolicy,
    gamma = gamma,
    firstVisitOnly = firstVisitOnly,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Implements the Incremental Monte Carlo Control algorithm, a reinforcement learning algorithm
 * that uses a Monte Carlo method to estimate the action-value function and improves the policy incrementally.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions in the environment.
 * @param initialPolicy The initial policy to be used for decision-making, represented as a Q-function-based policy.
 * @param gamma The discount factor for future rewards, must be in the range [0, 1]. Default value is 0.99.
 * @param alpha A schedule for the learning rate used to update the Q-function. Defaults to a constant schedule of 0.05.
 * @param firstVisitOnly If true, only the first visit to a state-action pair in a trajectory is considered
 *                       for updating the Q-function. If false, all visits are considered. Defaults to true.
 * @param onQFunctionUpdate A callback invoked after each Q-function update, allowing interventions or logging.
 *                          Default is an empty lambda.
 * @param onPolicyUpdate A callback invoked after each policy update, allowing interventions or logging.
 *                       Default is an empty lambda.
 * @return An implementation of the `TrajectoryQFunctionAlgorithm` interface, which can be used to run
 *         and manage the Incremental Monte Carlo Control algorithm.
 */
fun <State, Action> incrementalMonteCarloControl(
    initialPolicy: QFunctionPolicy<State, Action>,
    gamma: Double = 0.99,
    alpha: ParameterSchedule = constantParameterSchedule(0.05),
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): TrajectoryQFunctionAlgorithm<State, Action> = IncrementalMonteCarloControl(
    initialPolicy = initialPolicy,
    gamma = gamma,
    alpha = alpha,
    firstVisitOnly = firstVisitOnly,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Implements an off-policy Monte Carlo control algorithm for reinforcement learning. This method estimates
 * the optimal action-value function and improves the target policy using trajectories collected under a
 * behavioral policy.
 *
 * @param behavioralPolicy The policy used to generate trajectories. It may behave differently from the targetPolicy.
 * @param targetPolicy The policy being improved during the learning process.
 * @param gamma Discount factor for future rewards. Must be in the range [0, 1]. Defaults to 0.99.
 * @param onQFunctionUpdate Callback invoked when the action-value function (Q-function) is updated.
 * @param onPolicyUpdate Callback invoked when the target policy is updated.
 * @return An instance of TrajectoryQFunctionAlgorithm representing the off-policy Monte Carlo control process.
 */
fun <State, Action> offPolicyMonteCarloControl(
    behavioralPolicy: QFunctionPolicy<State, Action>,
    targetPolicy: QFunctionPolicy<State, Action>,
    gamma: Double = 0.99,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): TrajectoryQFunctionAlgorithm<State, Action> = OffPolicyMonteCarloControl(
    behavioralPolicy = behavioralPolicy,
    targetPolicy = targetPolicy,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Represents off-policy control strategies in reinforcement learning where two policies are used:
 * one for generating behavior (behavioral policy) and another for evaluating or improving (target policy).
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @property behavioralPolicy The policy used to generate behavior or actions during interactions with the environment.
 * @property targetPolicy The policy being evaluated or improved to optimize performance.
 */
data class OffPolicyControls<State, Action>(
    val behavioralPolicy: QFunctionPolicy<State, Action>,
    val targetPolicy: QFunctionPolicy<State, Action>
)

/**
 * Constructs an off-policy control mechanism using an epsilon-greedy target policy
 * and an epsilon-soft behavioral policy. The target policy is used for evaluation
 * or improvement, while the behavioral policy is used for generating actions during
 * environment interaction.
 *
 * @param Q The Q-function that maps state-action pairs to their corresponding
 *          action-value estimates.
 * @param stateActions A mapping of all possible actions available for each state
 *                     within the environment.
 * @param targetEpsilon A schedule defining the epsilon parameter for the epsilon-greedy
 *                      target policy. This controls the level of exploration vs. exploitation.
 * @param behaviorEpsilon A schedule defining the epsilon parameter for the epsilon-soft
 *                        behavioral policy. This controls the level of exploration
 *                        performed during interactions.
 * @param rng An optional random number generator for controlling randomness in action
 *            selection by the policies. Defaults to Random.Default.
 * @return An OffPolicyControls object containing both the behavioral and target policies.
 */
fun <State, Action> epsilonGreedySoftOffPolicyControls(
    Q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    targetEpsilon: ParameterSchedule,
    behaviorEpsilon: ParameterSchedule,
    rng: Random = Random.Default
): OffPolicyControls<State, Action> {

    val targetPolicy = epsilonGreedyPolicy(
        Q = Q,
        stateActions = stateActions,
        epsilon = targetEpsilon,
        rng = rng
    )
    val behavioralPolicy = epsilonSoftPolicy(
        Q = Q,
        stateActions = stateActions,
        epsilon = behaviorEpsilon,
        rng = rng
    )
    return OffPolicyControls(
        behavioralPolicy = behavioralPolicy,
        targetPolicy = targetPolicy
    )
}

/**
 * Implements the Q-Learning algorithm to optimize a policy based on observed transitions and rewards.
 *
 * @param initialPolicy The initial Q-function policy which determines the action-selection strategy.
 * @param alpha A parameter schedule defining the learning rate for Q-function updates.
 * @param gamma The discount factor, which determines the importance of future rewards.
 * @param onQFunctionUpdate A callback invoked after updating the Q-function for added extensibility. The default is no-op.
 * @param onPolicyUpdate A callback invoked after updating the policy for added extensibility. The default is no-op.
 * @return An instance of TransitionQFunctionAlgorithm which encapsulates the Q-Learning algorithm.
 */
fun <State, Action> qLearning(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): TransitionQFunctionAlgorithm<State, Action> = QLearning(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)

/**
 * Implements the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm.
 * SARSA is an on-policy temporal-difference control algorithm that updates the action-value
 * function based on the current action and the observed successor state-action pair.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions in the environment.
 * @param initialPolicy The initial policy defined by a Q-function, determining which
 * actions to take in given states.
 * @param alpha A schedule for the learning rate. This determines how much new information
 * overrides old information during the update to the Q-function.
 * @param gamma The discount factor for future rewards, determining the relative importance
 * of immediate versus later rewards.
 * @param onQFunctionUpdate A callback triggered each time the Q-function is updated. This
 * can be used for monitoring or logging purposes.
 * @param onPolicyUpdate A callback triggered each time the policy is updated, allowing for
 * optional hooks or side effects during policy updates.
 * @return A SARSA-based implementation of a transition Q-function algorithm.
 */
fun <State, Action> sarsa(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): TransitionQFunctionAlgorithm<State, Action> = SARSA(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)

/**
 * Creates and configures an Expected SARSA algorithm for reinforcement learning.
 *
 * @param initialPolicy The initial Q-function policy that guides the selection of actions.
 * @param alpha The learning rate represented as a parameter schedule.
 * @param gamma The discount factor, which determines the importance of future rewards.
 * @param onQFunctionUpdate Callback invoked when the Q-function is updated. Default is an empty callback.
 * @param onPolicyUpdate Callback invoked when the policy is updated. Default is an empty callback.
 * @return A configured instance of the Expected SARSA algorithm.
 */
fun <State, Action> expectedSarsa(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): TransitionQFunctionAlgorithm<State, Action> = ExpectedSARSA(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Implements the n-step SARSA (State-Action-Reward-State-Action) algorithm for reinforcement learning.
 * This algorithm updates the policy based on n-step temporal difference errors.
 *
 * @param initialPolicy The initial policy, represented as a Q-function policy, to be updated during learning.
 * @param alpha The learning rate schedule that controls the step size for value updates.
 * @param gamma The discount factor, which determines the importance of future rewards.
 * @param n The number of steps used in the n-step update process.
 * @param onQFunctionUpdate A callback that executes upon updating the Q-function.
 * @param onPolicyUpdate A callback that executes upon updating the policy.
 * @return An instance of TrajectoryQFunctionAlgorithm used to execute the n-step SARSA algorithm.
 */
fun <State, Action> nStepSarsa(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): TrajectoryQFunctionAlgorithm<State, Action> = NStepSARSA(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    n = n,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)