package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable
import kotlin.random.*

fun <State, Action> randomPolicy(
    actionProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(actionProvider, rng)

fun <State, Action> greedyPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>,
    qTable: QTable<State, Action>
): Policy<State, Action> = GreedyPolicy(stateActionListProvider, qTable)

fun <State, Action> epsilonGreedyPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>,
    explorationFactor: ExplorationFactor,
    qTable: QTable<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = EpsilonGreedyPolicy( stateActionListProvider, qTable, explorationFactor, rng)

fun <State, Action> softMaxPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>,
    temperature: ExplorationFactor,
    qTable: QTable<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = SoftmaxPolicy(
    stateActionListProvider = stateActionListProvider,
    qTable = qTable,
    temperature = temperature,
    rng = rng
)

fun <State, A> deterministicPolicy(
    map: Map<State, A>
): Policy<State, A> = Policy { map[it]!! }

fun <State, Action> epsilonSoftPolicy(
    actions: StateActionListProvider<State, Action>,
    qTable: QTable<State, Action>,
    epsilon: ExplorationFactor,
    rng: Random = Random.Default
): Policy<State, Action> = EpsilonSoftPolicy(actions, qTable, epsilon, rng)

fun constantEpsilon(factor: Double) = ExplorationFactor { factor }

fun decayingEpsilon(factor: Double, decayRate: Double, minFactor: Double    ): ExplorationFactor {
    var epsilon = factor
    return ExplorationFactor {
        epsilon = (epsilon * decayRate).coerceAtLeast(minFactor)
        epsilon
    }
}
