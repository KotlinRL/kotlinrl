package io.github.kotlinrl.core

import io.github.kotlinrl.core.algorithms.defaultStateActionKeyFunction
import kotlin.random.Random

typealias LearningAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.LearningAlgorithm<State, Action>
typealias QFunctionAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.QFunctionAlgorithm<State, Action>
typealias QTable = io.github.kotlinrl.core.algorithms.QTable
typealias VTable = io.github.kotlinrl.core.algorithms.VTable
typealias StateActionKey<State, Action> = io.github.kotlinrl.core.algorithms.StateActionKey<State, Action>
typealias ComparableIntList = io.github.kotlinrl.core.algorithms.ComparableIntList
typealias BellmanValueFunctionIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanValueFunctionIteration<State, Action>
typealias BellmanQFunctionIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanQFunctionIteration<State, Action>
typealias BellmanPolicyIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.BellmanPolicyIteration<State, Action>
typealias StateActionKeyFunction<State, Action> = io.github.kotlinrl.core.algorithms.StateActionKeyFunction<State, Action>
typealias StateKeyFunction<State> = io.github.kotlinrl.core.algorithms.StateKeyFunction<State>
typealias OnPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OnPolicyMonteCarloControl<State, Action>
typealias IncrementalMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.IncrementalMonteCarloControl<State, Action>
typealias OffPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OffPolicyMonteCarloControl<State, Action>
typealias ExpectedSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.ExpectedSARSA<State, Action>
typealias QLearning<State, Action> = io.github.kotlinrl.core.algorithms.td.QLearning<State, Action>
typealias SARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.SARSA<State, Action>
typealias NStepSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.nstep.NStepSARSA<State, Action>
typealias TabularTDAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.td.TabularTDAlgorithm<State, Action>
typealias DPIteration<State, Action> = io.github.kotlinrl.core.algorithms.dp.DPIteration<State, Action>
typealias MCAlgorithm<State, Action> = io.github.kotlinrl.core.algorithms.mc.MonteCarloAlgorithm<State, Action>
typealias ProbabilisticTransition<State, Action> = io.github.kotlinrl.core.algorithms.dp.ProbabilisticTransition<State, Action>
typealias ProbabilisticTrajectory<State, Action> = List<ProbabilisticTransition<State, Action>>
typealias EmpiricalMDPModel<State, Action> = io.github.kotlinrl.core.algorithms.dp.EmpiricalMDPModel<State, Action>
typealias MDPModel<State, Action> = io.github.kotlinrl.core.algorithms.dp.MDPModel<State, Action>
typealias DPValueFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.dp.DPValueFunctionEstimator<State, Action>
typealias DPQFunctionEstimator<State, Action> = io.github.kotlinrl.core.algorithms.dp.DPQFunctionEstimator<State, Action>

fun <State, Action> bellmanValueFunctionIteration(
    initialV: EnumerableValueFunction<State>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActionListProvider: StateActionListProvider<State, Action>,
    onValueFunctionUpdate: (EnumerableValueFunction<State>) -> Unit = { }
): DPIteration<State, Action> = BellmanValueFunctionIteration(
    initialV = initialV,
    model = EmpiricalMDPModel(env, initialV.allStates(), numSamples),
    gamma = gamma,
    theta = theta,
    stateActionListProvider = stateActionListProvider,
    onValueFunctionUpdate = onValueFunctionUpdate
)

fun <State, Action> bellmanQFunctionIteration(
    initialQ: EnumerableQFunction<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActionListProvider: StateActionListProvider<State, Action>,
    onQFunctionUpdate: (EnumerableQFunction<State, Action>) -> Unit = { }
): DPIteration<State, Action> = BellmanQFunctionIteration(
    initialQ = initialQ,
    model = EmpiricalMDPModel(env, initialQ.allStates(), numSamples),
    gamma = gamma,
    theta = theta,
    stateActionListProvider = stateActionListProvider,
    onQFunctionUpdate = onQFunctionUpdate
)

fun <State, Action> bellmanPolicyIteration(
    initialV: EnumerableValueFunction<State>,
    initialPolicy: Policy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    numSamples: Int = 100,
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateActionListProvider: StateActionListProvider<State, Action>,
    onValueFunctionUpdate: (ValueFunction<State>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
): DPIteration<State, Action> = BellmanPolicyIteration(
    initialPolicy = initialPolicy,
    initialV = initialV,
    model = EmpiricalMDPModel(env, initialV.allStates(), numSamples),
    gamma = gamma,
    theta = theta,
    stateActionListProvider = stateActionListProvider,
    onValueFunctionUpdate = onValueFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)

fun <State, Action> onPolicyMonteCarloControl(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    gamma: Double,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
): MCAlgorithm<State, Action> = OnPolicyMonteCarloControl(
    initialPolicy = initialPolicy,
    initialQ = initialQ,
    improvement = improvement,
    gamma = gamma,
    firstVisitOnly = firstVisitOnly,
    stateActionKeyFunction = stateActionKeyFunction,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

fun <State, Action> incrementalMonteCarloControl(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    gamma: Double = 0.99,
    alpha: ParameterSchedule = constantParameterSchedule(0.05),
    firstVisitOnly: Boolean = true,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
): MCAlgorithm<State, Action> = IncrementalMonteCarloControl(
    initialPolicy = initialPolicy,
    initialQ = initialQ,
    improvement = improvement,
    gamma = gamma,
    alpha = alpha,
    firstVisitOnly = firstVisitOnly,
    stateActionKeyFunction = stateActionKeyFunction,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

fun <State, Action> offPolicyMonteCarloControl(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    gamma: Double = 0.99,
    targetPolicy: Policy<State, Action>,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
): MCAlgorithm<State, Action> = OffPolicyMonteCarloControl(
    initialPolicy = initialPolicy,
    initialQ = initialQ,
    improvement = improvement,
    gamma = gamma,
    targetPolicy = targetPolicy,
    stateActionKeyFunction = stateActionKeyFunction,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

data class OffPolicyControls<State, Action>(
    val behavioralPolicy: StochasticPolicy<State, Action>,
    val targetPolicy: QFunctionPolicy<State, Action>
)

fun <State, Action> epsilonGreedySoftOffPolicyControls(
    q: QFunction<State, Action>,
    stateActionListProvider: StateActionListProvider<State, Action>,
    targetEpsilon: ParameterSchedule,
    behaviorEpsilon: ParameterSchedule,
    rng: Random = Random.Default
): OffPolicyControls<State, Action> {

    val targetPolicy = epsilonGreedyPolicy(
        q = q,
        epsilon = targetEpsilon,
        stateActionListProvider = stateActionListProvider,
        rng = rng
    )
    val behavioralPolicy = epsilonSoftPolicy(
        q = q,
        epsilon = behaviorEpsilon,
        stateActionListProvider = stateActionListProvider,
        rng = rng
    )
    return OffPolicyControls(
        behavioralPolicy = behavioralPolicy,
        targetPolicy = targetPolicy
    )
}

fun <State, Action> qLearning(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    alpha: ParameterSchedule,
    gamma: Double
): TabularTDAlgorithm<State, Action> = QLearning(initialPolicy, initialQ, improvement, onQFunctionUpdate, onPolicyUpdate, alpha, gamma)

fun <State, Action> sarsa(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    alpha: ParameterSchedule,
    gamma: Double
): TabularTDAlgorithm<State, Action> = SARSA(initialPolicy, initialQ, improvement, onQFunctionUpdate, onPolicyUpdate, alpha, gamma)

fun <State, Action> expectedSarsa(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    alpha: ParameterSchedule,
    gamma: Double,
    stateActionListProvider: StateActionListProvider<State, Action>
): TabularTDAlgorithm<State, Action> = ExpectedSARSA(
    initialPolicy = initialPolicy,
    initialQ = initialQ,
    improvement = improvement,
    alpha = alpha,
    gamma = gamma,
    stateActionListProvider = stateActionListProvider,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)

fun <State, Action> nStepSarsa(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    stateActionListProvider: StateActionListProvider<State, Action>
): TabularTDAlgorithm<State, Action> = NStepSARSA(
    initialPolicy = initialPolicy,
    initialQ = initialQ,
    improvement = improvement,
    alpha = alpha,
    gamma = gamma,
    n = n,
    stateActionListProvider = stateActionListProvider,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate,
)