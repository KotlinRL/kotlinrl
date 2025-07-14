package io.github.kotlinrl.core

import kotlin.random.*

typealias ExplorationFactor = io.github.kotlinrl.core.policy.ExplorationFactor
typealias RandomPolicy<State, Action> = io.github.kotlinrl.core.policy.RandomPolicy<State, Action>
typealias GreedyPolicy = io.github.kotlinrl.core.policy.GreedyPolicy
typealias EpsilonGreedyPolicy = io.github.kotlinrl.core.policy.EpsilonGreedyPolicy
typealias SoftmaxPolicy = io.github.kotlinrl.core.policy.SoftmaxPolicy
typealias EpsilonSoftPolicy = io.github.kotlinrl.core.policy.EpsilonSoftPolicy
typealias Policy<State, Action> = io.github.kotlinrl.core.policy.Policy<State, Action>
typealias ProbabilisticPolicy<State, Action> = io.github.kotlinrl.core.policy.ProbabilisticPolicy<State, Action>
typealias PolicyProbabilities<State, Action> = io.github.kotlinrl.core.policy.PolicyProbabilities<State, Action>
typealias StateActionListProvider<State, Action> = io.github.kotlinrl.core.policy.StateActionListProvider<State, Action>
typealias MutablePolicy<State, Action> = io.github.kotlinrl.core.policy.MutablePolicy<State, Action>

fun <State, Action> randomPolicy(
    actionProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(actionProvider, rng)

fun greedyPolicy(
    qTable: QTable
): Policy<IntArray, Int> = GreedyPolicy(qTable)

fun epsilonGreedyPolicy(
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    explorationFactor: ExplorationFactor,
    qTable: QTable,
    rng: Random = Random.Default
): Policy<IntArray, Int> = EpsilonGreedyPolicy(stateActionListProvider, qTable, explorationFactor, rng)

fun softMaxPolicy(
    temperature: ExplorationFactor,
    qTable: QTable,
    rng: Random = Random.Default
): Policy<IntArray, Int> = SoftmaxPolicy(
    qTable = qTable,
    temperature = temperature,
    rng = rng
)

fun epsilonSoftPolicy(
    actions: StateActionListProvider<IntArray, Int>,
    qTable: QTable,
    epsilon: ExplorationFactor,
    rng: Random = Random.Default
): Policy<IntArray, Int> = EpsilonSoftPolicy(
    stateActionListProvider = actions,
    qTable=qTable,
    epsilon = epsilon,
    rng = rng)

fun constantEpsilon(factor: Double) = ExplorationFactor { factor }

fun decayingEpsilon(factor: Double, decayRate: Double, minFactor: Double): ExplorationFactor {
    var epsilon = factor
    return ExplorationFactor {
        epsilon = (epsilon * decayRate).coerceAtLeast(minFactor)
        epsilon
    }
}
