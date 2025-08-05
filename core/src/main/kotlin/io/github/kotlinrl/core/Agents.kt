package io.github.kotlinrl.core

//import io.github.kotlinrl.core.algorithms.defaultStateActionKeyFunction
import io.github.kotlinrl.core.agent.Transition
import java.util.*

typealias Agent<State, Action> = io.github.kotlinrl.core.agent.Agent<State, Action>
typealias Transition<State, Action> = Transition<State, Action>
typealias TransitionObserver<State, Action> = io.github.kotlinrl.core.agent.TransitionObserver<State, Action>
typealias TrajectoryObserver<State, Action> = io.github.kotlinrl.core.agent.TrajectoryObserver<State, Action>
typealias PolicyAgent<State, Action> = io.github.kotlinrl.core.agent.PolicyAgent<State, Action>
typealias LearningAgent<State, Action> = io.github.kotlinrl.core.agent.LearningAgent<State, Action>

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

fun <State, Action> learningAgent(
    id: String = UUID.randomUUID().toString(),
    algorithm: LearningAlgorithm<State, Action>,
): Agent<State, Action> = LearningAgent(
    id = id,
    algorithm = algorithm,
)

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

@Suppress("UNCHECKED_CAST")
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

