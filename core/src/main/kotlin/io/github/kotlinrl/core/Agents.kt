package io.github.kotlinrl.core

import java.util.*

/**
 * A type alias for the `Agent` interface, representing an abstraction for agents interacting with environments
 * in a reinforcement learning setup. This alias serves to simplify usage references within the primary codebase.
 *
 * An agent observes its environment's state, decides on actions based on its policy or logic, and has the ability
 * to adapt or learn from feedback like state transitions or trajectories. This abstraction is foundational for
 * implementing reinforcement learning solutions.
 *
 * @param State The type parameter representing the state space of the environment.
 * @param Action The type parameter representing the action space of the environment.
 */
typealias Agent<State, Action> = io.github.kotlinrl.core.agent.Agent<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.agent.Transition`, representing a single step
 * interaction between an agent and the environment in reinforcement learning.
 *
 * This alias is used to simplify the reference to the `Transition` class, which provides
 * structured information about the state, action, reward, next state, and termination
 * details of a transition within an environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias Transition<State, Action> = io.github.kotlinrl.core.agent.Transition<State, Action>
/**
 * Type alias for `TransitionObserver`, representing a functional interface used to observe
 * state-action transitions in reinforcement learning environments.
 *
 * Provides a mechanism for receiving and processing transition events, enabling actions such
 * as logging, learning updates, or customizing behaviors during transitions.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias TransitionObserver<State, Action> = io.github.kotlinrl.core.agent.TransitionObserver<State, Action>
/**
 * A type alias for the `TrajectoryObserver` functional interface from the KotlinRL library.
 *
 * This alias provides a shorthand for referencing the observer, responsible for processing
 * trajectories in reinforcement learning environments. It observes sequences of transitions
 * during an episode, including state-action interactions, rewards, and resulting states.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias TrajectoryObserver<State, Action> = io.github.kotlinrl.core.agent.TrajectoryObserver<State, Action>
/**
 * A type alias representing an agent that follows a specific policy for decision-making
 * in a reinforcement learning environment.
 *
 * The `PolicyAgent` type encapsulates logic for selecting actions based on states, observing
 * transitions, and managing complete trajectories. This alias simplifies the reference to
 * the `PolicyAgent` class within the codebase.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias PolicyAgent<State, Action> = io.github.kotlinrl.core.agent.PolicyAgent<State, Action>
/**
 * Represents a type alias for the `LearningAgent` class from the `io.github.kotlinrl.core.agent` package.
 *
 * A `LearningAgent` is an abstraction for agents in reinforcement learning systems that can
 * interact with an environment, learn from feedback, and adapt their behavior over time.
 *
 * @param State The type that defines the state space of the environment in which the agent operates.
 * @param Action The type that defines the action space of the agent to interact with the environment.
 */
typealias LearningAgent<State, Action> = io.github.kotlinrl.core.agent.LearningAgent<State, Action>

/**
 * Creates a policy-based agent that selects actions based on a given policy and optionally observes
 * transitions and trajectories in a reinforcement learning environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID string.
 * @param policy The policy the agent uses to select actions based on observed states.
 * @param onTransition An optional callback invoked for each individual state-action transition.
 *                     Defaults to an empty observer that performs no action.
 * @param onTrajectory An optional callback invoked for processing trajectories (sequences of transitions)
 *                     across episodes. Defaults to an empty observer that performs no action.
 * @return An instance of the agent with the specified policy and observation capabilities.
 */
fun <State, Action> policyAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    onTransition: TransitionObserver<State, Action> = TransitionObserver { },
    onTrajectory: TrajectoryObserver<State, Action> = TrajectoryObserver { _, _ -> }
): Agent<State, Action> = PolicyAgent(
    id = id,
    policy = policy,
    onTransition = onTransition,
    onTrajectory = onTrajectory,
)

/**
 * Creates a learning agent capable of interacting with an environment
 * and adapting its behavior based on the specified learning algorithm.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id The unique identifier of the learning agent. If not provided, a randomly generated UUID is used.
 * @param algorithm The learning algorithm used by the agent to determine actions and update its behavior.
 * @return A new learning agent instance using the specified learning algorithm and identifier.
 */
fun <State, Action> learningAgent(
    id: String = UUID.randomUUID().toString(),
    algorithm: LearningAlgorithm<State, Action>,
): Agent<State, Action> = LearningAgent(
    id = id,
    algorithm = algorithm,
)

/**
 * Creates an agent utilizing the Bellman Value Function Iteration algorithm for planning in a
 * reinforcement learning environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID string.
 * @param initialV The initial value function over the state space, used for iterative computation.
 * @param env The model-based environment that simulates the state transitions and dynamics.
 * @param numSamples The number of samples to draw from the environment for estimating transition probabilities. Defaults to 100.
 * @param gamma The discount factor that determines the importance of future rewards. Defaults to 0.99.
 * @param theta The threshold for convergence in the value function updates. Defaults to 1e-6.
 * @param stateActions A function that maps each state to its possible actions.
 * @param onValueFunctionUpdate A callback function invoked upon every update of the value function.
 *                              Defaults to an empty function with no operation.
 * @return An agent that uses the value function-based policy derived from the Bellman Value Function Iteration algorithm for action selection.
 */
fun <State, Action> bellmanValueFunctionIterationAgent(
    id: String = UUID.randomUUID().toString(),
    initialV: EnumerableValueFunction<State>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
): Agent<State, Action> = policyAgent(
    id = id,
    policy = bellmanValueFunctionIteration(
        initialV = initialV,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onValueFunctionUpdate = onValueFunctionUpdate,
    ).plan()
)

/**
 * Creates an agent utilizing the Bellman Q-function iteration process for reinforcement learning.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID string.
 * @param initialQ The initial Q-function providing estimates for state-action values.
 * @param env The model-based environment the agent interacts with.
 * @param numSamples The number of samples from the environment used during estimation. Defaults to 100.
 * @param gamma The discount factor (0 ≤ gamma ≤ 1) determining the trade-off between short-term
 *        and long-term rewards. Defaults to 0.99.
 * @param theta The convergence threshold for iterative updates, representing the minimum change
 *        required to halt the update process. Defaults to 1e-6.
 * @param stateActions A function to retrieve the list of valid actions for a given state.
 * @param onQFunctionUpdate Callback invoked whenever the Q-function is updated during the process.
 *        Defaults to an empty observer.
 * @return An agent based on a policy derived from the Bellman Q-function iteration process.
 */
fun <State, Action> bellmanQFunctionIterationAgent(
    id: String = UUID.randomUUID().toString(),
    initialQ: EnumerableQFunction<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: (EnumerableQFunction<State, Action>) -> Unit = { }
): Agent<State, Action> = policyAgent(
    id = id,
    policy = bellmanQFunctionIteration(
        initialQ = initialQ,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onQFunctionUpdate = onQFunctionUpdate,
    ).plan()
)

/**
 * Creates an agent that uses the Bellman Policy Iteration algorithm to optimize its policy
 * for decision-making in a reinforcement learning environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID string.
 * @param initialV The initial enumerable value function used to estimate state values.
 * @param initialPolicy The initial policy used by the agent for decision-making.
 * @param env The model-based environment where the agent operates.
 * @param numSamples The number of samples used to approximate state transitions in the empirical environment model. Defaults to 100.
 * @param gamma The discount factor, determining the relative importance of future rewards. Defaults to 0.99.
 * @param theta A small threshold to determine the convergence of the value function. Defaults to 1e-6.
 * @param stateActions A function that maps states to their possible actions.
 * @param onValueFunctionUpdate A callback function invoked when the value function is updated.
 * @param onPolicyUpdate A callback function invoked when the policy is updated.
 * @return An agent implementing the Bellman Policy Iteration algorithm for reinforcement learning.
 */
fun <State, Action> bellmanPolicyIterationAgent(
    id: String = UUID.randomUUID().toString(),
    initialV: EnumerableValueFunction<State>,
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): Agent<State, Action> = policyAgent(
    id = id,
    policy = bellmanPolicyIteration(
        initialV = initialV,
        initialPolicy = initialPolicy,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onValueFunctionUpdate = onValueFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
    ).plan()
)

/**
 * Creates an agent that follows the on-policy Monte Carlo control algorithm to update
 * both its Q-function and policy based on observed episodes. This method constructs
 * the agent with the algorithm and settings provided.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id The unique identifier of the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy that guides the agent's actions. It is represented
 * as a Q-function-based policy.
 * @param gamma The discount factor in the range [0, 1], determining the importance of
 * future rewards relative to immediate rewards.
 * @param firstVisitOnly If true, only the first visit to a state in an episode
 * is considered when updating the Q-function. If false, every visit to a state
 * contributes to the Q-function update.
 * @param onQFunctionUpdate Callback invoked when the Q-function is updated, providing
 * an opportunity to observe or modify its behavior.
 * @param onPolicyUpdate Callback invoked when the policy is updated, providing
 * an opportunity to observe or modify its behavior.
 * @return An agent implementing the on-policy Monte Carlo control algorithm, capable
 * of learning and interacting with an environment.
 */
fun <State, Action> onPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = onPolicyMonteCarloControl(
        initialPolicy = initialPolicy,
        gamma = gamma,
        firstVisitOnly = firstVisitOnly,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
    )
)

/**
 * Creates an off-policy Monte Carlo control agent with specified behavioral and target policies and additional configurations.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id The unique identifier for the agent. Defaults to a randomly generated UUID if not provided.
 * @param behavioralPolicy The exploratory behavioral policy used to generate trajectories in the environment.
 * @param targetPolicy The deterministic or fixed target policy associated with the agent, updated based on the Q-function.
 * @param gamma The discount factor applied to future rewards. Defaults to 0.99.
 * @param onQFunctionUpdate A callback invoked upon updates to the Q-function during learning. Defaults to an empty lambda.
 * @param onPolicyUpdate A callback invoked upon updates to the policy during learning. Defaults to an empty lambda.
 * @return An agent configured for off-policy Monte Carlo control learning with the specified parameters.
 */
fun <State, Action> offPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    behavioralPolicy: QFunctionPolicy<State, Action>,
    targetPolicy: QFunctionPolicy<State, Action>,
    gamma: Double = 0.99,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = offPolicyMonteCarloControl(
        behavioralPolicy = behavioralPolicy,
        gamma = gamma,
        targetPolicy = targetPolicy,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
    )
)

/**
 * Creates an incremental Monte Carlo control agent for reinforcement learning.
 *
 * This function constructs an agent that utilizes the incremental Monte Carlo control algorithm
 * to optimize its behavior based on collected experiences. The agent uses a policy initialized
 * with the provided Q-function policy and adjusts it iteratively while learning from the environment.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id An optional unique identifier for the agent. If not provided, a random UUID will be generated.
 * @param initialPolicy The initial Q-function policy used by the agent to select actions.
 * @param gamma The discount factor for future rewards, where values closer to 1 consider distant rewards more heavily.
 * @param alpha A schedule defining the learning rate used when updating the Q-function. Defaults to a constant value of 0.05.
 * @param firstVisitOnly Specifies whether to use only the first visit to a state-action pair for updates (if true)
 *                       or all visits (if false). Defaults to true.
 * @param onQFunctionUpdate A callback invoked whenever the Q-function is updated. Receives arguments related to the update.
 * @param onPolicyUpdate A callback invoked whenever the policy is updated. Receives arguments related to the update.
 * @return An agent configured to use the incremental Monte Carlo control algorithm for adapting its policy.
 */
fun <State, Action> incrementalMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    gamma: Double = 0.99,
    alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = incrementalMonteCarloControl(
        initialPolicy = initialPolicy,
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate
    )
)

/**
 * Creates a Q-Learning-based reinforcement learning agent.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id The unique identifier of the agent. Defaults to a randomly generated UUID if not provided.
 * @param initialPolicy The initial Q-function policy used by the agent to determine actions.
 * @param alpha The learning rate parameter schedule for the Q-Learning algorithm.
 * @param gamma The discount factor for future rewards in the Q-Learning algorithm.
 * @param onQFunctionUpdate Callback executed on updates to the Q-function.
 * @param onPolicyUpdate Callback executed on updates to the policy.
 * @return A Q-Learning-based agent capable of interacting with an environment and adapting its behavior.
 */
fun <State, Action> qLearningAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = qLearning(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate
    )
)

/**
 * Creates a SARSA agent capable of interacting with an environment and learning based on the
 * SARSA reinforcement learning algorithm.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id The unique identifier for the agent. If not provided, a random UUID will be generated.
 * @param initialPolicy The initial policy used by the agent, defined as a Q-function policy over states and actions.
 * @param alpha A schedule for the learning rate, which controls the weight of new information in updates.
 * @param gamma The discount factor, which determines the importance of future rewards.
 * @param onQFunctionUpdate Callback function executed after every Q-function update.
 * @param onPolicyUpdate Callback function executed after every policy update.
 * @return An agent instance that uses the SARSA algorithm for learning and decision-making.
 */
fun <State, Action> sarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = sarsa(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate
    )
)

/**
 * Creates an agent using the Expected SARSA algorithm for reinforcement learning.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id The unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy used in the Expected SARSA algorithm, represented as a Q-function policy.
 * @param alpha A schedule for the learning rate parameter, which determines how fast the agent adjusts based on new experiences.
 * @param gamma The discount factor, which represents the importance of future rewards over immediate rewards.
 * @param onQFunctionUpdate A callback triggered when the Q-function is updated, allowing for optional custom functionality.
 * @param onPolicyUpdate A callback triggered when the policy is updated, allowing for optional custom functionality.
 * @return An agent that uses the Expected SARSA algorithm for learning and decision-making.
 */
fun <State, Action> expectedSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = expectedSarsa(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate
    )
)

/**
 * Creates an n-step SARSA learning agent that utilizes the given policy and parameters to
 * learn and adapt its behavior over time.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param id The unique identifier of the learning agent. Defaults to a randomly generated UUID if not provided.
 * @param initialPolicy The initial policy used by the agent, often represented as a Q-function policy.
 * @param alpha A parameter schedule that determines the learning rate for Q-function updates.
 * @param gamma The discount factor used to balance immediate and future rewards, must be in the range [0, 1].
 * @param n The number of steps to accumulate rewards before applying updates, influencing the learning process.
 * @param onQFunctionUpdate A callback function executed on Q-function updates, allowing custom behavior during learning.
 * @param onPolicyUpdate A callback function executed on policy updates, allowing for custom behavior when the policy changes.
 * @return A learning agent implementing the n-step SARSA algorithm.
 */
fun <State, Action> nStepSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = nStepSarsa(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
        n = n,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate
    )
)

