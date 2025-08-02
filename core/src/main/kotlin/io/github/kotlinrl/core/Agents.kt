package io.github.kotlinrl.core

import io.github.kotlinrl.core.agent.Transition
import io.github.kotlinrl.core.algorithms.*
import io.github.kotlinrl.core.algorithms.StateActionKeyFunction
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
    stateActionListProvider: StateActionListProvider<State, Action>,
    onValueFunctionUpdate: (EnumerableValueFunction<State>) -> Unit = { }
): Agent<State, Action> = policyAgent(
    id = id,
    policy = bellmanValueFunctionIteration(
        initialV = initialV,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActionListProvider = stateActionListProvider,
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
    stateActionListProvider: StateActionListProvider<State, Action>,
    onQFunctionUpdate: (EnumerableQFunction<State, Action>) -> Unit = { }
): Agent<State, Action> = policyAgent(
    id = id,
    policy = bellmanQFunctionIteration(
        initialQ = initialQ,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActionListProvider = stateActionListProvider,
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
    stateActionListProvider: StateActionListProvider<State, Action>,
    onValueFunctionUpdate: (ValueFunction<State>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
): Agent<State, Action> = policyAgent(
    id = id,
    policy = bellmanPolicyIteration(
        initialV = initialV,
        initialPolicy = initialPolicy,
        env = env,
        numSamples = numSamples,
        gamma = gamma,
        theta = theta,
        stateActionListProvider = stateActionListProvider,
        onValueFunctionUpdate = onValueFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
    ).plan()
)

@Suppress("UNCHECKED_CAST")
fun <State, Action> onPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    gamma: Double,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    firstVisitOnly: Boolean = true
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = onPolicyMonteCarloControl(
        initialPolicy = initialPolicy,
        initialQ = initialPolicy.q,
        improvement = initialPolicy as PolicyImprovementStrategy<State, Action>,
        gamma = gamma,
        firstVisitOnly = firstVisitOnly,
        stateActionKeyFunction = stateActionKeyFunction,
        onQFunctionUpdate = onQFunctionUpdate,
    )
)

@Suppress("UNCHECKED_CAST")
fun <State, Action> offPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    gamma: Double = 0.99,
    targetPolicy: Policy<State, Action>,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = offPolicyMonteCarloControl(
        initialPolicy = initialPolicy,
        initialQ = initialQ,
        improvement = targetPolicy as PolicyImprovementStrategy<State, Action>,
        gamma = gamma,
        targetPolicy = targetPolicy,
        stateActionKeyFunction = stateActionKeyFunction,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate,
    )
)

@Suppress("UNCHECKED_CAST")
fun <State, Action> incrementalMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    gamma: Double = 0.99,
    alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    firstVisitOnly: Boolean = true,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = incrementalMonteCarloControl(
        initialPolicy = initialPolicy,
        initialQ = initialPolicy.q,
        improvement = initialPolicy as PolicyImprovementStrategy<State, Action>,
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly,
        stateActionKeyFunction = stateActionKeyFunction,
        onQFunctionUpdate = onQFunctionUpdate,
        onPolicyUpdate = onPolicyUpdate
    )
)

@Suppress("UNCHECKED_CAST")
fun <State, Action> qLearningAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: QFunctionPolicy<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    alpha: ParameterSchedule,
    gamma: Double
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = qLearning(
        initialPolicy = initialPolicy,
        initialQ = initialPolicy.q,
        improvement = initialPolicy as PolicyImprovementStrategy<State, Action>,
        alpha = alpha,
        gamma = gamma,
        onQFunctionUpdate = onQFunctionUpdate,
    )
)

@Suppress("UNCHECKED_CAST")
fun <State, Action> sarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    alpha: ParameterSchedule,
    gamma: Double
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = sarsa(
        initialPolicy = initialPolicy,
        initialQ = initialQ,
        improvement = initialPolicy as PolicyImprovementStrategy<State, Action>,
        alpha = alpha,
        gamma = gamma,
        onQFunctionUpdate = onQFunctionUpdate,
    )
)


@Suppress("UNCHECKED_CAST")
fun <State, Action> expectedSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    stateActionListProvider: StateActionListProvider<State, Action>
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = expectedSarsa(
        initialPolicy = initialPolicy,
        initialQ = initialQ,
        improvement = initialPolicy as PolicyImprovementStrategy<State, Action>,
        alpha = alpha,
        gamma = gamma,
        stateActionListProvider = stateActionListProvider
    )
)

@Suppress("UNCHECKED_CAST")
fun <State, Action> nStepSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    stateActionListProvider: StateActionListProvider<State, Action>
): Agent<State, Action> = learningAgent(
    id = id,
    algorithm = nStepSarsa(
        initialPolicy = initialPolicy,
        initialQ = initialQ,
        improvement = initialPolicy as PolicyImprovementStrategy<State, Action>,
        alpha = alpha,
        gamma = gamma,
        n = n,
        stateActionListProvider = stateActionListProvider
    )
)

