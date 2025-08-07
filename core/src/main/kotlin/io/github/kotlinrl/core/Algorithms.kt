package io.github.kotlinrl.core

import kotlin.random.*

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
typealias LearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.LearningAlgorithm<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.algorithms.base.HybridPolicyPlanningAlgorithm`.
 *
 * Represents a hybrid policy planning algorithm that combines model-based and model-free
 * reinforcement learning approaches. This type alias simplifies references to the
 * fully qualified class name within the codebase, enabling cleaner and more concise
 * type declarations.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the possible actions in the environment.
 */
typealias HybridPolicyPlanningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.HybridPolicyPlanningAlgorithm<State, Action>
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
typealias TransitionLearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.TransitionLearningAlgorithm<State, Action>
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
typealias TrajectoryLearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.TrajectoryLearningAlgorithm<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.algorithms.base.EstimateQ_fromTransition`.
 *
 * Represents a functional interface for estimating an updated Q-function from a given
 * state-action-reward-next-state transition. It provides a mechanism to modify or recalculate a
 * Q-function to better approximate the quality of state-action pairs during reinforcement learning.
 *
 * This alias simplifies the reference to the `EstimateQ_fromTransition` type within the library
 * or application code.
 *
 * @param State The type representing the states of the environment.
 * @param Action The type representing the actions performable in the environment.
 */
typealias EstimateQ_fromTransition<State, Action> = io.github.kotlinrl.core.algorithms.base.EstimateQ_fromTransition<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.algorithms.base.EstimateQ_fromTrajectory`.
 *
 * Represents a functional interface intended for estimating a new Q-function
 * based on a trajectory of state-action-reward transitions. This interface
 * defines the logic or algorithm for improving Q-function evaluation using
 * complete episodes to refine state-action value estimations, valuable in
 * reinforcement learning contexts.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions taken within the environment.
 */
typealias EstimateQ_fromTrajectory<State, Action> = io.github.kotlinrl.core.algorithms.base.EstimateQ_fromTrajectory<State, Action>
/**
 * A type alias for the `EstimateQ_fromProbabilisticTrajectory` functional interface.
 *
 * It represents a mechanism to estimate or update a Q-function using state-action
 * probabilities from a probabilistic trajectory in reinforcement learning.
 *
 * The alias simplifies reference to the `io.github.kotlinrl.core.algorithms.base.EstimateQ_fromProbabilisticTrajectory`
 * interface, which is used for iterative learning and updates to a Q-function based
 * on stochastic transitions.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be performed in the environment.
 */
typealias EstimateQ_fromProbabilisticTrajectory<State, Action> = io.github.kotlinrl.core.algorithms.base.EstimateQ_fromProbabilisticTrajectory<State, Action>
/**
 * Represents a type alias for the `EstimateV_fromTrajectory` functional interface.
 *
 * This alias simplifies referencing the `io.github.kotlinrl.core.algorithms.base.EstimateV_fromTrajectory`
 * interface, which defines a mechanism to estimate a value function (V) based on a given trajectory
 * of states and actions. The functional interface processes an initial value function and updates it
 * using the information from the trajectory, enabling value function estimation in reinforcement learning algorithms.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be performed in the environment.
 */
typealias EstimateV_fromTrajectory<State, Action> = io.github.kotlinrl.core.algorithms.base.EstimateV_fromTrajectory<State, Action>
/**
 * Type alias for the `io.github.kotlinrl.core.algorithms.base.EstimateV_fromTransition` functional interface.
 *
 * Represents a strategy used to estimate or update the value function within reinforcement learning contexts
 * based on observed state-action transitions and the current value function.
 *
 * This alias simplifies references in the codebase, providing a concise way to describe a
 * mechanism for recalculating the value function in response to environment dynamics.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing actions performed in the environment.
 */
typealias EstimateV_fromTransition<State, Action> = io.github.kotlinrl.core.algorithms.base.EstimateV_fromTransition<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.algorithms.base.EstimateV_fromProbabilisticTrajectory`.
 *
 * This alias defines a functional interface for estimating a value function (V) in the context of
 * reinforcement learning, based on a probabilistic trajectory. The probabilistic trajectory represents
 * sequences of states, actions, and their associated probabilities, which reflect the dynamics of
 * the environment. By utilizing this trajectory, implementations can refine or update the value
 * function using methods such as Monte Carlo evaluation or Temporal-Difference learning.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions available in the environment.
 */
typealias EstimateV_fromProbabilisticTrajectory<State, Action> = io.github.kotlinrl.core.algorithms.base.EstimateV_fromProbabilisticTrajectory<State, Action>
/**
 * A type alias for the `PolicyPlanner` interface, simplifying its reference within the codebase.
 *
 * `PolicyPlanner` defines a contract for creating policies in the context of reinforcement learning,
 * leveraging Q-functions, environment models, and state-action mappings for Markov Decision Processes (MDPs).
 *
 * This alias makes it easier to utilize the `PolicyPlanner` in creating or manipulating agent policies
 * based on specific environment dynamics and decision-making criteria.
 *
 * @param State the type that represents the possible states in the environment.
 * @param Action the type that represents the possible actions in the environment.
 */
typealias PolicyPlanner<State, Action> = io.github.kotlinrl.core.algorithms.base.PolicyPlanner<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.algorithms.dp.BellmanIterateV`.
 *
 * Represents an iterative policy improvement algorithm leveraging the Bellman equations
 * to refine value and Q-functions for a given Markov Decision Process (MDP). It applies
 * successive Bellman updates to estimate optimal policies by updating value functions
 * until convergence criteria are met.
 *
 * This alias provides a simplified reference to the `BellmanIterateV` class within the
 * relevant codebase, allowing for improved readability and usability.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions available in the environment.
 */
typealias BellmanIterateV<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanIterateV<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.algorithms.dp.BellmanIterateQ`.
 *
 * Represents an iterative approach to refining Q-values using the Bellman equation until convergence.
 * The purpose of this alias is to simplify and improve code readability when referring to the
 * BellmanIterateQ class within the reinforcement learning library.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the allowable actions in the environment.
 */
typealias BellmanIterateQ<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanIterateQ<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.algorithms.dp.BellmanIteratePolicy`.
 *
 * Represents the Bellman policy iteration algorithm applied to a Markov Decision Process (MDP).
 * This algorithm alternates between policy evaluation and policy improvement steps to compute
 * the optimal policy for a given MDP. It leverages Bellman equations, probabilistic trajectories,
 * and iterative updates of value and Q-functions to ensure convergence towards an optimal policy.
 *
 * Provides a more concise reference to `io.github.kotlinrl.core.algorithms.dp.BellmanIteratePolicy`.
 *
 * @param State The type parameter representing states in the MDP.
 * @param Action The type parameter representing actions in the MDP.
 */
typealias BellmanIteratePolicy<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanIteratePolicy<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.algorithms.mc.OnPolicyMonteCarloControl`.
 *
 * This type alias provides a simplified reference to the `OnPolicyMonteCarloControl` class,
 * which implements an on-policy Monte Carlo control algorithm for reinforcement learning.
 *
 * The alias is parameterized with `State` and `Action` types, representing the agent's
 * state space and action space, respectively. The underlying class leverages Monte Carlo
 * methods to estimate the Q-function and improve the policy based on sampled trajectories.
 *
 * This alias is useful to streamline references and improve code readability within the
 * codebase that utilizes Monte Carlo-based control methods.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions in the environment.
 */
typealias OnPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OnPolicyMonteCarloControl<State, Action>
/**
 * Typealias for `io.github.kotlinrl.core.algorithms.mc.IncrementalMonteCarloControl`.
 *
 * Encapsulates the implementation of an incremental Monte Carlo control algorithm
 * for reinforcement learning, which combines policy iteration with value function
 * estimation. It utilizes complete episodes for refining the policy and updating
 * the Q-function incrementally based on observed returns.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be taken in the environment.
 */
typealias IncrementalMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.IncrementalMonteCarloControl<State, Action>
/**
 * Type alias for the `OffPolicyMonteCarloControl` class in the `io.github.kotlinrl.core.algorithms.mc` package.
 *
 * Provides a shorthand reference to the implementation of the Off-Policy Monte Carlo Control algorithm
 * for reinforcement learning. This type alias simplifies the usage of the class by reducing the verbosity
 * of referencing it in the codebase.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be taken in the environment.
 */
typealias OffPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OffPolicyMonteCarloControl<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.algorithms.td.classic.ExpectedSARSA`.
 *
 * Represents the Expected SARSA algorithm for reinforcement learning, which is an on-policy
 * algorithm that updates Q-values based on the expected rewards and transitions, integrating
 * the probabilities of all possible actions in the next state under the given policy. This
 * approach reduces variance compared to traditional SARSA while maintaining stability and
 * efficiency in learning.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions that can be executed within the environment.
 */
typealias ExpectedSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.ExpectedSARSA<State, Action>
/**
 * Typealias for `io.github.kotlinrl.core.algorithms.td.classic.QLearning`.
 *
 * Simplifies the reference to the implementation of the Q-Learning algorithm, an
 * off-policy reinforcement learning method. Q-Learning enables learning an optimal
 * policy by iteratively updating the action-value function (Q-function) based on
 * observed state-action transitions and rewards.
 *
 * This alias facilitates concise use of the Q-Learning implementation within the
 * codebase while retaining its full functionality and configuration options, such
 * as learning rates, discount factors, and policy updates.
 *
 * @param State Represents the type for states in the environment.
 * @param Action Represents the type for actions within the environment.
 */
typealias QLearning<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.QLearning<State, Action>
/**
 * A type alias for the `SARSA` class from the `io.github.kotlinrl.core.algorithms.td.classic` package.
 *
 * This alias represents the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm,
 * an on-policy temporal difference learning method used to improve policies based on observed experiences
 * in an environment. SARSA updates its Q-function by following the policy being actively used during learning,
 * and balances future and immediate rewards through a specified discount factor.
 *
 * The parameters `State` and `Action` define the state and action spaces of the environment, respectively.
 */
typealias SARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.SARSA<State, Action>
/**
 * Typealias for `io.github.kotlinrl.core.algorithms.td.nstep.NStepSARSA`.
 *
 * Represents an implementation of the n-step SARSA algorithm, which is a
 * temporal difference (TD) reinforcement learning method for policy
 * evaluation and improvement. This typealias provides a shorthand
 * reference to the n-step SARSA class for applications involving n-step
 * state-action-reward-state-action (SARSA) updates.
 *
 * The algorithm focuses on accumulating rewards over a sequence of n steps
 * and updating the Q-function based on the temporal difference error
 * calculated over the trajectory. It operates in an on-policy setting,
 * where the policy being evaluated and improved is also used to generate
 * the action sequence.
 *
 * The n-step SARSA implementation is configurable with various parameters
 * including learning rate, discount factor, trajectory length, and callback
 * functions for observing updates to the Q-function and policy.
 *
 * @param State The type representing the state space in the environment.
 * @param Action The type representing the action space in the environment.
 */
typealias NStepSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.nstep.NStepSARSA<State, Action>
/**
 * A type alias for a function that updates a Q-function.
 *
 * Represents an operation performed on a `QFunction` instance.
 * This alias is used as a shorthand for defining a function signature
 * that takes a `QFunction<State, Action>` as input and performs an update
 * operation without returning a value.
 *
 * @param State The type representing the states of the environment.
 * @param Action The type representing the actions available in the environment.
 */
typealias QFunctionUpdate<State, Action> = (QFunction<State, Action>) -> Unit
/**
 * A type alias for a function that updates a `ValueFunction` instance for a given state type.
 *
 * This provides a concise way to define operations that modify or transform a value function
 * during reinforcement learning or optimization processes. Typically, the function takes a
 * `ValueFunction` as an input and applies some updates to it, potentially based on a policy,
 * learning algorithm, or dynamic adjustment.
 *
 * @param State The type representing the state space of the `ValueFunction`.
 */
typealias ValueFunctionUpdate<State> = (ValueFunction<State>) -> Unit


/**
 * Implements a Bellman value function iteration algorithm for hybrid policy planning. This method
 * integrates model-based reinforcement learning with value iteration to optimize policies. The
 * algorithm iterates over the value function using the Bellman equation and updates the policy
 * accordingly.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions within the environment.
 * @param initialPolicy the policy to initialize the learning process with.
 * @param env the model-based environment that provides state transitions and rewards.
 * @param numSamples the number of samples used to approximate transitions in the environment. Defaults to 100.
 * @param gamma the discount factor for future rewards. Must be in the range [0, 1]. Defaults to 0.99.
 * @param theta the threshold for convergence in value function updates. Defaults to 1e-6.
 * @param stateActions a function mapping states to their available actions.
 * @param onQFunctionUpdate a callback invoked when the Q-function is updated.
 * @param onPolicyUpdate a callback invoked when the policy is updated.
 * @param onValueFunctionUpdate a callback invoked when the value function is updated.
 * @return an instance of the `LearningAlgorithm` that applies the Bellman value function iteration.
 */
fun <State, Action> bellmanValueFunctionIteration(
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onValueFunctionUpdate: ValueFunctionUpdate<State> = { },
): LearningAlgorithm<State, Action> = HybridPolicyPlanningAlgorithm(
    initialPolicy = initialPolicy,
    model = EmpiricalMDPModel(
        env = env,
        allStates = env.allStates(),
        allActions = env.allStates().flatMap { stateActions(it) }.toList(),
        numSamples = numSamples
    ),
    stateActions = stateActions,
    policyPlanner = BellmanIterateV(
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onValueFunctionUpdate = onValueFunctionUpdate,
    ),
    onPolicyUpdate = onPolicyUpdate,
    onQFunctionUpdate = onQFunctionUpdate
)

/**
 * Performs the Bellman Q-function iteration for a given policy and environment. This function
 * integrates model-based and iterative approaches within a hybrid policy planning algorithm.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the possible actions within the environment.
 * @param initialPolicy The initial policy used to begin the Bellman Q-function iteration.
 * @param env The model-based environment in which the algorithm operates.
 * @param numSamples The number of samples to use for estimating transitions and rewards. Defaults to 100.
 * @param gamma The discount factor determining the significance of future rewards. Defaults to 0.99.
 * @param theta The convergence threshold for Q-function updates. Iteration stops when updates are smaller than this value. Defaults to 1e-6.
 * @param stateActions A function that defines the valid actions available for each state.
 * @param onQFunctionUpdate A callback function invoked whenever the Q-function is updated. Defaults to an empty callback function.
 * @param onPolicyUpdate A callback function invoked whenever the policy is updated. Defaults to an empty callback function.
 * @return A learning algorithm that iteratively updates the Q-function and corresponding policy using the Bellman equation.
 */
fun <State, Action> bellmanQFunctionIteration(
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): LearningAlgorithm<State, Action> = HybridPolicyPlanningAlgorithm(
    initialPolicy = initialPolicy,
    model = EmpiricalMDPModel(
        env = env,
        allStates = env.allStates(),
        allActions = env.allStates().flatMap { stateActions(it) }.toList(),
        numSamples = numSamples
    ),
    stateActions = stateActions,
    policyPlanner = BellmanIterateQ(
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onQFunctionUpdate = onQFunctionUpdate,
    ),
    onPolicyUpdate = onPolicyUpdate,
    onQFunctionUpdate = onQFunctionUpdate
)

/**
 * Implements a hybrid policy iteration algorithm using the Bellman update approach.
 * This function integrates model-based reinforcement learning techniques to iteratively update the policy
 * and value function for a given environment.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the possible actions within the environment.
 * @param initialPolicy the initial policy provided as the starting point for policy iteration.
 * @param env the model-based environment used for simulating steps and generating the empirical MDP model.
 * @param numSamples the number of samples used to approximate the transition dynamics of the model.
 *                   A higher value increases accuracy but also computational cost. Default is 100.
 * @param gamma the discount factor (between 0 and 1) used to weigh future rewards relative to immediate rewards. Default is 0.99.
 * @param theta the threshold used to determine convergence of the value function update. Default is 1e-6.
 * @param stateActions a function mapping each state to the list of possible actions in that state.
 * @param onQFunctionUpdate an optional callback function triggered on each Q-function update. Default is an empty function.
 * @param onPolicyUpdate an optional callback function triggered whenever the policy is recalculated. Default is an empty function.
 * @param onValueFunctionUpdate a callback function triggered on each value function update during policy iteration.
 * @return a learning algorithm instance that integrates the Bellman-based policy iteration with empirical MDP modeling.
 */
fun <State, Action> bellmanPolicyIteration(
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onValueFunctionUpdate: ValueFunctionUpdate<State>
): LearningAlgorithm<State, Action> = HybridPolicyPlanningAlgorithm(
    initialPolicy = initialPolicy,
    model = EmpiricalMDPModel(
        env = env,
        allStates = env.allStates(),
        allActions = env.allStates().flatMap { stateActions(it) }.toList(),
        numSamples = numSamples
    ),
    stateActions = stateActions,
    policyPlanner = BellmanIteratePolicy(
        initialPolicy = initialPolicy,
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onValueFunctionUpdate = onValueFunctionUpdate,
    ),
    onPolicyUpdate = onPolicyUpdate,
    onQFunctionUpdate = onQFunctionUpdate
)

/**
 * Creates an implementation of the On-Policy Monte Carlo Control learning algorithm.
 *
 * The function uses On-Policy Monte Carlo methods to improve a policy and estimate
 * the Q-function based on episodes sampled from the environment. It supports both
 * first-visit and every-visit Monte Carlo approaches and provides hooks for updates
 * to the Q-function and policy.
 *
 * @param State The type representing the state's space of the environment.
 * @param Action The type representing the action's space of the environment.
 * @param initialPolicy The initial policy used to determine the agent's action-selection
 *        strategy at the beginning of learning.
 * @param gamma The discount factor used to weigh future rewards relative to immediate rewards.
 *        Should be in the range [0, 1].
 * @param firstVisitOnly A flag indicating whether the first-visit approach (true) or the every-visit
 *        approach (false) should be used for updating the Q-function. Default is true.
 * @param onQFunctionUpdate A callback function invoked whenever the Q-function is updated.
 *        Default is an empty callback.
 * @param onPolicyUpdate A callback function invoked whenever the policy is updated.
 *        Default is an empty callback.
 * @return A constructed instance of the On-Policy Monte Carlo Control learning algorithm
 *         that can be used for reinforcement learning in an environment.
 */
fun <State, Action> onPolicyMonteCarloControl(
    initialPolicy: Policy<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): LearningAlgorithm<State, Action> = OnPolicyMonteCarloControl(
    initialPolicy = initialPolicy,
    gamma = gamma,
    firstVisitOnly = firstVisitOnly,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Implements the incremental Monte Carlo control algorithm for policy improvement and
 * value estimation in reinforcement learning. This method refines the policy iteratively
 * by processing complete episodes, using state-action trajectories to estimate Q-values and updating the policy.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be taken in the environment.
 * @param initialPolicy The initial policy that guides the agent's behavior in the environment.
 * @param gamma The discount factor for future rewards. Default is 0.99.
 * @param alpha A parameter schedule defining the learning rate for Q-value updates. Defaults to a constant schedule with a value of 0.05.
 * @param firstVisitOnly If true, Q-values are updated only for the first occurrence of state-action pairs in a trajectory. Default is true.
 * @param onQFunctionUpdate A callback executed after each Q-function update, allowing for custom behavior or monitoring during the learning process.
 * @param onPolicyUpdate A callback executed after each policy update, enabling additional custom actions or monitoring.
 * @return A reinforcement learning algorithm instance that can be used to train policies using the incremental Monte Carlo control method.
 */
fun <State, Action> incrementalMonteCarloControl(
    initialPolicy: Policy<State, Action>,
    gamma: Double = 0.99,
    alpha: ParameterSchedule = constantParameterSchedule(0.05),
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): LearningAlgorithm<State, Action> = IncrementalMonteCarloControl(
    initialPolicy = initialPolicy,
    gamma = gamma,
    alpha = alpha,
    firstVisitOnly = firstVisitOnly,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Implements the Off-Policy Monte Carlo Control algorithm for reinforcement learning.
 * This function uses trajectories generated by a behavioral policy to improve a target policy
 * through off-policy Q-function updates with importance sampling.
 *
 * @param State Type representing the states in the environment.
 * @param Action Type representing the actions that can be taken in the environment.
 * @param behavioralPolicy The policy used to generate trajectories during exploration.
 *                         This is the policy actively interacting with the environment.
 * @param targetPolicy The policy being optimized, which learns an improved mapping
 *                     of states to actions based on the Q-function updates.
 * @param gamma The discount factor, a value in the range [0.0, 1.0], determining the
 *              weight of future rewards in Q-function updates. Default is 0.99.
 * @param onQFunctionUpdate A callback function that is invoked after each Q-function update,
 *                          allowing for custom operations or logging. Default is no-op.
 * @param onPolicyUpdate A callback function that is invoked when the target policy is updated.
 *                       This provides a mechanism for responding to policy improvements. Default is no-op.
 * @return A configured instance of `OffPolicyMonteCarloControl` as the learning algorithm
 *         for optimizing the target policy.
 */
fun <State, Action> offPolicyMonteCarloControl(
    behavioralPolicy: Policy<State, Action>,
    targetPolicy: Policy<State, Action>,
    gamma: Double = 0.99,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): LearningAlgorithm<State, Action> = OffPolicyMonteCarloControl(
    behavioralPolicy = behavioralPolicy,
    targetPolicy = targetPolicy,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Represents a structure for off-policy reinforcement learning control mechanisms.
 *
 * This class is used to define and manage the relationship between two policies:
 * a behavioral policy and a target policy. The behavioral policy is used to generate
 * trajectories or samples by interacting with the environment, whereas the target policy
 * is the policy being optimized or evaluated.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @property behavioralPolicy The policy used to determine actions during interaction with the environment.
 * @property targetPolicy The policy being optimized or used to compute target values.
 */
data class OffPolicyControls<State, Action>(
    val behavioralPolicy: Policy<State, Action>,
    val targetPolicy: Policy<State, Action>
)

/**
 * Constructs an off-policy control mechanism using epsilon-greedy and epsilon-soft strategies
 * for target and behavioral policies, respectively. This method is utilized in reinforcement
 * learning scenarios where exploration and exploitation are balanced through epsilon-controlled
 * mechanisms.
 *
 * @param Q The Q-function representing the value estimation for state-action pairs.
 * @param stateActions A function that provides the available actions for a given state.
 * @param targetEpsilon A parameter schedule controlling the exploration rate (epsilon) for the target policy.
 * @param behaviorEpsilon A parameter schedule controlling the exploration rate (epsilon) for the behavioral policy.
 * @param rng A random number generator for stochastic action selection, defaulting to `Random.Default`.
 * @return An `OffPolicyControls` instance containing the target and behavioral policies, respectively.
 */
fun <State, Action> epsilonGreedySoftOffPolicyControls(
    Q: QFunction<State, Action>,
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
 * Creates a Q-Learning-based learning algorithm for reinforcement learning tasks.
 *
 * Q-Learning is an off-policy, model-free reinforcement learning algorithm that updates
 * the action-value function (Q-function) to approximate the optimal policy. It uses
 * the Temporal Difference (TD) update rule with the formula:
 *
 * Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) − Q(s, a)]
 *
 * This implementation allows customization of the learning process through parameter
 * schedules, event callbacks for Q-function updates, and updates to the policy.
 *
 * @param State the type representing the environment's states.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param initialPolicy the starting policy that governs action selection in the environment.
 * @param alpha a parameter schedule that controls the learning rate for Q-function updates.
 * @param gamma the discount factor, a value between 0 and 1, that determines the weight of future rewards.
 * @param onQFunctionUpdate a callback triggered whenever the Q-function is updated.
 * @param onPolicyUpdate a callback triggered whenever the policy is updated.
 * @return a Q-Learning-based learning algorithm for state-action value estimation.
 */
fun <State, Action> qLearning(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): LearningAlgorithm<State, Action> = QLearning(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)

/**
 * Implements the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm.
 *
 * This method creates a SARSA learning algorithm that updates Q-values based on the current policy
 * and observed transitions in an environment. It provides flexibility through callbacks for
 * Q-function and policy updates, customization of the learning rate, and control over the
 * discount factor.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions in the environment.
 * @param initialPolicy the initial policy the agent should follow; this policy is updated during learning.
 * @param alpha a [ParameterSchedule] specifying the learning rate, which may change over time for fine-tuning convergence.
 * @param gamma the discount factor, a value between 0 and 1 indicating the importance of future rewards relative to immediate rewards.
 * @param onQFunctionUpdate a callback function invoked whenever the Q-function is updated, allowing monitoring or logging of updates.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated, enabling external actions on policy changes.
 * @return a [LearningAlgorithm] instance representing the configured SARSA algorithm, ready for interaction with an environment.
 */
fun <State, Action> sarsa(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): LearningAlgorithm<State, Action> = SARSA(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)

/**
 * Creates an instance of the Expected SARSA reinforcement learning algorithm.
 * Expected SARSA is an on-policy algorithm that calculates the Q-value updates
 * by considering the expected value of the next state over all possible actions,
 * reducing variance and improving learning stability.
 *
 * @param initialPolicy The initial policy used by the agent to make decisions.
 * @param alpha A schedule specifying the learning rate to update Q-values over time.
 * @param gamma The discount factor representing the importance of future rewards, constrained between 0 and 1.
 * @param onQFunctionUpdate A callback invoked after updating the Q-function, allowing additional processing or monitoring.
 * @param onPolicyUpdate A callback invoked after the policy update, enabling additional handling or monitoring of changes.
 * @return A reinforcement learning algorithm implementing Expected SARSA with specified parameters and callbacks.
 */
fun <State, Action> expectedSarsa(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): LearningAlgorithm<State, Action> = ExpectedSARSA(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

/**
 * Creates an instance of the n-step SARSA reinforcement learning algorithm for policy evaluation
 * and improvement. This method configures the algorithm with the provided parameters and callbacks.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param initialPolicy The initial policy governing the agent's action selection. This policy
 *                       will be updated iteratively as learning progresses.
 * @param alpha A schedule defining the learning rate (step size) for Q-function updates. The value
 *              determines the magnitude of updates to the Q-function based on temporal difference errors.
 * @param gamma The discount factor applied to future rewards during the learning process. This parameter
 *              balances the importance of immediate versus long-term rewards. Should be in the range [0, 1].
 * @param n The number of steps used in the computation of n-step returns. This defines the length of
 *          the trajectory considered for temporal difference updates.
 * @param onQFunctionUpdate A callback function invoked after each Q-function update. This can be used
 *                          for monitoring or logging the updates to the Q-function.
 * @param onPolicyUpdate A callback function invoked after each policy update. Enables tracking or logging
 *                       the changes to the policy during the learning process.
 * @return An instance of `LearningAlgorithm` configured with the n-step SARSA implementation.
 */
fun <State, Action> nStepSarsa(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): LearningAlgorithm<State, Action> = NStepSARSA(
    initialPolicy = initialPolicy,
    alpha = alpha,
    gamma = gamma,
    n = n,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)
