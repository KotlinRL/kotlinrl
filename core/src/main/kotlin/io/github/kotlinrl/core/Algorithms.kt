package io.github.kotlinrl.core

import io.github.kotlinrl.core.env.ModelBasedEnv
import java.util.UUID
import kotlin.IntArray

typealias MonteCarloControl = io.github.kotlinrl.core.algorithms.mc.MonteCarloControl
typealias ExpectedSARSA = io.github.kotlinrl.core.algorithms.td.ExpectedSARSA
typealias QLearning = io.github.kotlinrl.core.algorithms.td.QLearning
typealias SARSA = io.github.kotlinrl.core.algorithms.td.SARSA
typealias QTable = io.github.kotlinrl.core.algorithms.QTable
typealias VTable = io.github.kotlinrl.core.algorithms.VTable
typealias PTable = io.github.kotlinrl.core.algorithms.PTable
typealias ValueIteration = io.github.kotlinrl.core.algorithms.dp.ValueIteration
typealias PolicyIteration = io.github.kotlinrl.core.algorithms.dp.PolicyIteration

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
) : ExpectedSARSA = ExpectedSARSA(
    qTable = qTable,
    alpha = alpha,
    gamma = gamma,
    stateActionListProvider = stateActionListProvider,
    policyProbabilities = policyProbabilities
)

fun valueIteration(
    env: ModelBasedEnv
): Policy<IntArray, Int> = valueIteration(
    size = env.size,
    goal = env.goal,
    allActions = env::stateActionList,
    transition = env::nextState,
    reward = env::computeReward,
)

fun valueIteration(
    size: Int,
    goal: IntArray,
    allActions: StateActionListProvider<IntArray, Int>,
    transition: TransitionFunction<IntArray, Int>,
    reward: RewardFunction<IntArray, Int>
): Policy<IntArray, Int> {
    val planner = ValueIteration()
    return planner.plan(size, goal, allActions, transition, reward)
}

fun policyIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    env: ModelBasedEnv
): Policy<IntArray, Int> = policyIteration(
    gamma = gamma,
    theta = theta,
    size = env.size,
    goal = env.goal,
    allActions = env::stateActionList,
    transition = env::nextState,
    reward = env::computeReward,
)

fun policyIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    size: Int,
    goal: IntArray,
    allActions: StateActionListProvider<IntArray, Int>,
    transition: TransitionFunction<IntArray, Int>,
    reward: RewardFunction<IntArray, Int>
): Policy<IntArray, Int> {
    val planner = PolicyIteration(gamma, theta)
    return planner.plan(size, goal, allActions, transition, reward)
}