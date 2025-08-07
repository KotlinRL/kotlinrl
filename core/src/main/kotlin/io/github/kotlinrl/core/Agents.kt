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
 * Creates a learning agent that can interact with an environment, adapt its behavior through
 * feedback, and make decisions using a specified learning algorithm.
 *
 * @param id The unique identifier of the learning agent. Defaults to a randomly generated UUID.
 * @param algorithm The learning algorithm used by the agent to determine actions and update its behavior.
 * @return An instance of a learning agent configured with the specified identifier and algorithm.
 */
fun <State, Action> learningAgent(
    id: String = UUID.randomUUID().toString(),
    algorithm: LearningAlgorithm<State, Action>,
): Agent<State, Action> = LearningAgent(
    id = id,
    algorithm = algorithm
)

/**
 * Creates an agent that uses Bellman value function iteration to optimize its policy
 * in a model-based environment. The agent interacts with the environment, updates
 * the value function, policy, and Q-function iteratively, and supports callbacks for each update step.
 *
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy used by the agent, mapping states to action probabilities.
 * @param env The model-based environment in which the agent operates. Provides state transitions and rewards.
 * @param numSamples The number of samples used to empirically approximate transitions. Defaults to 100.
 * @param gamma The discount factor for future rewards, determining the balance between immediate and long-term rewards. Defaults to 0.99.
 * @param theta The threshold for convergence in value function updates. Iterations stop when the maximum change across states is less than this value. Defaults to 1e-6.
 * @param stateActions A function that maps states to the set of possible actions for each state.
 * @param onQFunctionUpdate A callback triggered after updates to the Q-function. Defaults to no-op.
 * @param onPolicyUpdate A callback triggered after updates to the policy. Defaults to no-op.
 * @param onValueFunctionUpdate A callback triggered after updates to the value function. Defaults to no-op.
 * @return An agent configured to perform Bellman value function iteration in the given environment.
 */
fun <State, Action> bellmanValueFunctionIterationAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onValueFunctionUpdate: ValueFunctionUpdate<State> = { },
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = bellmanValueFunctionIteration(
        initialPolicy = initialPolicy,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
        onValueFunctionUpdate = onValueFunctionUpdate,
    )
)

/**
 * Creates an agent that employs Bellman Q-function iteration for determining an optimal policy
 * in a model-based environment. The agent iterates over state-action pairs to update
 * its policy based on learned Q-values.
 *
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy for the agent, defining the initial behavior.
 * @param env The model-based environment enabling simulation of state transitions and rewards.
 * @param numSamples The number of samples used to approximate transitions and rewards in the environment. Defaults to 100.
 * @param gamma The discount factor for future rewards, typically a value between 0 and 1. Defaults to 0.99.
 * @param theta A convergence threshold indicating when the iteration process should terminate. Defaults to 1e-6.
 * @param stateActions A function that provides the possible actions for a given state.
 * @param onQFunctionUpdate A callback triggered when the Q-function is updated during the algorithm's execution. Defaults to an empty lambda.
 * @param onPolicyUpdate A callback triggered when the agent's policy is updated. Defaults to an empty lambda.
 * @return An agent configured with the Bellman Q-function iteration algorithm for policy optimization.
 */
fun <State, Action> bellmanQFunctionIterationAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = bellmanQFunctionIteration(
        initialPolicy = initialPolicy,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
    )
)

/**
 * Creates a reinforcement learning agent that applies Bellman's policy iteration algorithm to optimize its policy
 * within a model-based environment. The agent iteratively evaluates and improves its policy until convergence
 * based on the given parameters and callbacks.
 *
 * @param id The unique identifier of the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy used to start policy iteration. This policy will be iteratively improved.
 * @param env The model-based environment representing the problem dynamics, including state transitions and rewards.
 * @param numSamples The number of samples to approximate transitions and rewards for the environment. Default is 100.
 * @param gamma The discount factor for future rewards, controlling how the agent values future versus immediate rewards. Default is 0.99.
 * @param theta The convergence threshold for the policy evaluation process. Smaller values improve precision but increase computation. Default is 1e-6.
 * @param stateActions The function mapping states to their available actions.
 * @param onQFunctionUpdate A callback invoked whenever the Q-function is updated during learning. Default is an empty lambda.
 * @param onPolicyUpdate A callback invoked whenever the policy is updated during policy iteration. Default is an empty lambda.
 * @param onValueFunctionUpdate A callback invoked whenever the value function is updated during policy evaluation. Default is an empty lambda.
 * @return An agent configured to use Bellman's policy iteration algorithm for decision-making and policy optimization.
 */
fun <State, Action> bellmanPolicyIterationAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onValueFunctionUpdate: ValueFunctionUpdate<State> = { },
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = bellmanPolicyIteration(
        initialPolicy = initialPolicy,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActions = stateActions,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
        onValueFunctionUpdate = onValueFunctionUpdate,
    )
)

/**
 * Creates an agent that utilizes the on-policy Monte Carlo control method for reinforcement learning.
 * This agent follows a specific policy, collects rewards and state-action trajectories during interactions
 * with the environment, and uses the collected data to refine both the policy and the Q-function.
 *
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy used by the agent for decision-making, which will be updated during learning.
 * @param gamma The discount factor to apply to future rewards, where 0 ≤ gamma ≤ 1.
 * @param firstVisitOnly If true, only the first occurrence of a state-action pair in a trajectory is used for updates.
 *                       If false, all occurrences are used for updates. Defaults to true.
 * @param onQFunctionUpdate A callback invoked after every update to the Q-function, providing the updated values.
 * @param onPolicyUpdate A callback invoked after every update to the policy, providing the updated policy state.
 * @return An agent configured to implement the on-policy Monte Carlo control method for reinforcement learning.
 */
fun <State, Action> onPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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
 * Creates an off-policy Monte Carlo control learning agent. This agent estimates the optimal action-value
 * function using trajectories generated with a behavioral policy while improving a separate target policy.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be executed within the environment.
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param behavioralPolicy The policy used to generate trajectories during learning. This policy may differ from the target policy.
 * @param targetPolicy The policy being improved over time based on the action-value function.
 * @param gamma The discount factor for future rewards. Must be in the range [0, 1]. Defaults to 0.99.
 * @param onQFunctionUpdate Callback invoked upon an update to the action-value function (Q-function).
 * @param onPolicyUpdate Callback invoked when the target policy is updated.
 * @return An off-policy Monte Carlo control agent that implements the specified behavioral and target policies.
 */
fun <State, Action> offPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    behavioralPolicy: Policy<State, Action>,
    targetPolicy: Policy<State, Action>,
    gamma: Double = 0.99,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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
 * Creates an agent that uses the Incremental Monte Carlo Control algorithm to learn an action-value function and improve
 * its policy incrementally based on observed trajectories. The agent can perform exploration and updates while considering
 * first-visit or every-visit updates for the Q-function.
 *
 * @param id The unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy the agent uses for selecting actions, represented as a Q-function-based policy.
 * @param gamma The discount factor for future rewards. Must be within the range [0, 1]. Defaults to 0.99.
 * @param alpha A dynamically adjustable schedule for the learning rate used to update the Q-function. Defaults to 0.05 constant.
 * @param firstVisitOnly If true, only the first visit to a state-action pair in an episode is used to update the Q-function.
 *                       Defaults to true.
 * @param onQFunctionUpdate A callback function invoked after the Q-function is updated, useful for logging or custom behavior.
 * @param onPolicyUpdate A callback function invoked after the policy is updated, useful for logging or custom behavior.
 * @return An agent that learns and improves its policy through the Incremental Monte Carlo Control algorithm.
 */
fun <State, Action> incrementalMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    gamma: Double = 0.99,
    alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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
 * Creates a Q-learning-based agent for reinforcement learning tasks using the given parameters.
 *
 * @param id A unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy that the agent follows, which determines the action-selection strategy.
 * @param alpha A parameter schedule defining the learning rate for updating the Q-function.
 * @param gamma The discount factor, which determines the importance of future rewards relative to immediate rewards.
 * @param onQFunctionUpdate A callback triggered after the Q-function is updated. The default is an empty callback.
 * @param onPolicyUpdate A callback triggered after the policy is updated. The default is an empty callback.
 * @return An agent configured to use the Q-learning algorithm for learning and decision-making.
 */
fun <State, Action> qLearningAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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
 * Creates an agent that implements the SARSA (State-Action-Reward-State-Action) algorithm
 * for reinforcement learning. The SARSA algorithm is an on-policy method, where the action-value
 * function is updated based on the current action and the observed successor state-action pair.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions in the environment.
 * @param id The unique identifier for the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy defined by a Q-function, which determines the
 * actions to take in specific states.
 * @param alpha A schedule for the learning rate. This defines how much influence new information
 * has over past values when updating the Q-function.
 * @param gamma The discount factor for future rewards. It determines the relative importance
 * of immediate rewards versus future rewards.
 * @param onQFunctionUpdate A callback that is invoked each time the Q-function is updated. This
 * can be used for logging, monitoring, or additional side effects.
 * @param onPolicyUpdate A callback invoked each time the policy is updated. This allows hooks
 * or additional side effects during policy updates.
 * @return An agent configured to use the SARSA algorithm for learning and decision-making.
 */
fun <State, Action> sarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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
 * Creates an Expected SARSA learning agent for reinforcement learning tasks.
 *
 * The agent is initialized with a unique identifier, an initial policy for action selection,
 * a parameterized learning rate, and a discount factor for future rewards. Optional callbacks
 * can be provided to handle updates to the Q-function and the policy.
 *
 * @param id The unique identifier of the agent. Defaults to a randomly generated UUID string.
 * @param initialPolicy The initial policy guiding the selection of actions based on states.
 * @param alpha The parameter schedule for the learning rate, dictating the adjustment of Q-values.
 * @param gamma The discount factor, determining the weight of future rewards in decision-making.
 * @param onQFunctionUpdate Callback invoked when the Q-function is updated. Defaults to an empty callback.
 * @param onPolicyUpdate Callback invoked when the policy is updated. Defaults to an empty callback.
 * @return A reinforcement learning agent configured with the Expected SARSA algorithm.
 */
fun <State, Action> expectedSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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
 * Creates an n-step SARSA agent that interacts with an environment and learns using the n-step SARSA algorithm.
 *
 * @param id The unique identifier of the agent. Defaults to a randomly generated UUID.
 * @param initialPolicy The initial policy to guide the agent's decision-making, which will be updated during learning.
 * @param alpha The learning rate schedule governing updates to the Q-function.
 * @param gamma The discount factor, determining the importance of future rewards relative to immediate rewards.
 * @param n The number of steps used in the n-step update process during learning.
 * @param onQFunctionUpdate A callback invoked when the Q-function is updated.
 * @param onPolicyUpdate A callback invoked when the policy is updated.
 * @return An instance of an agent configured with the n-step SARSA learning algorithm.
 */
fun <State, Action> nStepSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
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

