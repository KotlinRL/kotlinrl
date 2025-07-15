package io.github.kotlinrl.core

typealias QTable = io.github.kotlinrl.core.algorithms.QTable
typealias VTable = io.github.kotlinrl.core.algorithms.VTable
typealias PTable = io.github.kotlinrl.core.algorithms.PTable
typealias ValueIteration = io.github.kotlinrl.core.algorithms.dp.ValueIteration
typealias PolicyIteration = io.github.kotlinrl.core.algorithms.dp.PolicyIteration
typealias OnPolicyMonteCarloControl = io.github.kotlinrl.core.algorithms.mc.OnPolicyMonteCarloControl
typealias ConstantAlphaMonteCarloControl = io.github.kotlinrl.core.algorithms.mc.ConstantAlphaMonteCarloControl
typealias OffPolicyMonteCarloControl = io.github.kotlinrl.core.algorithms.mc.OffPolicyMonteCarloControl
typealias ExpectedSARSA = io.github.kotlinrl.core.algorithms.td.ExpectedSARSA
typealias QLearning = io.github.kotlinrl.core.algorithms.td.QLearning
typealias SARSA = io.github.kotlinrl.core.algorithms.td.SARSA

fun qLearning(
    qTable: QTable,
    alpha: Double,
    gamma: Double
): QLearning = QLearning(qTable, alpha, gamma)

fun sarsa(
    qTable: QTable,
    alpha: Double,
    gamma: Double
): SARSA = SARSA(qTable, alpha, gamma)

fun expectedSARSA(
    qTable: QTable,
    alpha: Double,
    gamma: Double,
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    policyProbabilities: PolicyProbabilities<IntArray, Int>
): ExpectedSARSA = ExpectedSARSA(
    qTable = qTable,
    alpha = alpha,
    gamma = gamma,
    stateActionListProvider = stateActionListProvider,
    policyProbabilities = policyProbabilities
)

fun valueIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    env: ModelBasedEnv,
    stateShape: IntArray,
    stateActionListProvider: StateActionListProvider<IntArray, Int>
): Policy<IntArray, Int> = valueIterationPlanner(gamma, theta)
    .plan(
        stateShape = stateShape,
        stateActionListProvider = stateActionListProvider,
        transitionFunction = env::simulateStep
    )

fun valueIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateShape: IntArray,
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    transitionFunction: TransitionFunction<IntArray, Int>
): Policy<IntArray, Int> = valueIterationPlanner(gamma, theta)
    .plan(
        stateShape = stateShape,
        stateActionListProvider = stateActionListProvider,
        transitionFunction = transitionFunction
    )


fun policyIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    env: ModelBasedEnv,
    stateShape: IntArray,
    stateActionListProvider: StateActionListProvider<IntArray, Int>
): Policy<IntArray, Int> = policyIterationPlanner(gamma, theta)
    .plan(
        stateShape = stateShape,
        stateActionListProvider = stateActionListProvider,
        transitionFunction = env::simulateStep
    )

fun policyIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    stateShape: IntArray,
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    transitionFunction: TransitionFunction<IntArray, Int>
): Policy<IntArray, Int> = policyIterationPlanner(gamma, theta)
    .plan(
        stateShape = stateShape,
        stateActionListProvider = stateActionListProvider,
        transitionFunction = transitionFunction
    )


fun onPolicyMonteCarloControl(
    qTable: QTable,
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true
): EpisodeCallback<IntArray, Int> = OnPolicyMonteCarloControl(qTable, gamma, firstVisitOnly)

fun constantAlphaMonteCarloControl(
    qTable: QTable,
    gamma: Double = 0.99,
    alpha: Double = 0.05,
    firstVisitOnly: Boolean = true
): EpisodeCallback<IntArray, Int> = ConstantAlphaMonteCarloControl(qTable, gamma, alpha, firstVisitOnly)

fun offPolicyMonteCarloControl(
    qTable: QTable,
    gamma: Double = 0.99,
    behaviorPolicy: ProbabilisticPolicy<IntArray, Int>,
    targetPolicy: MutablePolicy<IntArray, Int>
): EpisodeCallback<IntArray, Int> = OffPolicyMonteCarloControl(
    qTable = qTable,
    gamma = gamma,
    behaviorPolicy = behaviorPolicy,
    targetPolicy = targetPolicy
)