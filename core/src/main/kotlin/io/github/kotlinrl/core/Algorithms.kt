package io.github.kotlinrl.core

import kotlin.random.*

typealias LearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.LearningAlgorithm<State, Action>
typealias QFunctionAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.QFunctionAlgorithm<State, Action>
typealias TDQError<State, Action> = io.github.kotlinrl.core.algorithms.td.TDQError<State, Action>
typealias TDVError<State, Action> = io.github.kotlinrl.core.algorithms.td.TDVError<State, Action>
typealias TDQErrors = io.github.kotlinrl.core.algorithms.td.TDQErrors
typealias TDVErrors = io.github.kotlinrl.core.algorithms.td.TDVErrors
typealias TransitionQFunctionAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.TransitionQFunctionAlgorithm<State, Action>
typealias TransitionQFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TransitionQFunctionEstimator<State, Action>
typealias TrajectoryQFunctionAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionAlgorithm<State, Action>
typealias TrajectoryQFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionEstimator<State, Action>
typealias TrajectoryValueFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TrajectoryValueFunctionEstimator<State, Action>
typealias TransitionValueFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.base.TransitionValueFunctionEstimator<State, Action>
typealias BellmanValueFunctionIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanValueFunctionIteration<State, Action>
typealias BellmanQFunctionIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanQFunctionIteration<State, Action>
typealias BellmanPolicyIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanPolicyIteration<State, Action>
typealias OnPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OnPolicyMonteCarloControl<State, Action>
typealias IncrementalMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.IncrementalMonteCarloControl<State, Action>
typealias OffPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OffPolicyMonteCarloControl<State, Action>
typealias ExpectedSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.ExpectedSARSA<State, Action>
typealias QLearning<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.QLearning<State, Action>
typealias SARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.classic.SARSA<State, Action>
typealias NStepSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.nstep.NStepSARSA<State, Action>
typealias DPIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.DPIteration<State, Action>
typealias DPValueFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.dp.DPValueFunctionEstimator<State, Action>
typealias EnumerableQFunctionUpdate<State, Action> = (EnumerableQFunction<State, Action>) -> Unit
typealias EnumerableValueFunctionUpdate<State> = (EnumerableValueFunction<State>) -> Unit


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

data class OffPolicyControls<State, Action>(
    val behavioralPolicy: QFunctionPolicy<State, Action>,
    val targetPolicy: QFunctionPolicy<State, Action>
)

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