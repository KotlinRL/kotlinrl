package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable
import kotlin.random.*

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
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    temperature: ExplorationFactor,
    qTable: QTable,
    rng: Random = Random.Default
): Policy<IntArray, Int> = SoftmaxPolicy(
    stateActionListProvider = stateActionListProvider,
    qTable = qTable,
    temperature = temperature,
    rng = rng
)

fun epsilonSoftPolicy(
    actions: StateActionListProvider<IntArray, Int>,
    qTable: QTable,
    epsilon: ExplorationFactor,
    rng: Random = Random.Default
): Policy<IntArray, Int> = EpsilonSoftPolicy(actions, qTable, epsilon, rng)

fun constantEpsilon(factor: Double) = ExplorationFactor { factor }

fun decayingEpsilon(factor: Double, decayRate: Double, minFactor: Double    ): ExplorationFactor {
    var epsilon = factor
    return ExplorationFactor {
        epsilon = (epsilon * decayRate).coerceAtLeast(minFactor)
        epsilon
    }
}
