package io.github.kotlinrl.core.policy

import kotlin.random.*

fun <State, Action> randomPolicy(
    actionProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(actionProvider, rng)

fun <State, Action> greedyPolicy(
    actionProvider: StateActionListProvider<State, Action>,
    Q: QFunction<State, Action>
): Policy<State, Action> = GreedyPolicy(actionProvider, Q)

fun <State, Action> epsilonGreedyPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>,
    explorationFactor: ExplorationFactor,
    Q: QFunction<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = EpsilonGreedyPolicy( stateActionListProvider, Q, explorationFactor)

fun <State, Action> softMaxPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>,
    temperature: ExplorationFactor,
    Q: QFunction<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = SoftmaxPolicy(
    stateActionListProvider = stateActionListProvider,
    Q = Q,
    temperature = temperature,
    rng = rng
)

fun <State, A> deterministicPolicy(
    map: Map<State, A>
): Policy<State, A> = Policy { map[it]!! }

fun <State, Action> epsilonSoftPolicy(
    actions: StateActionListProvider<State, Action>,
    Q: QFunction<State, Action>,
    epsilon: ExplorationFactor,
    rng: Random = Random.Default
): Policy<State, Action> = EpsilonSoftPolicy(actions, Q, epsilon, rng)

fun constantEpsilon(factor: Double) = ExplorationFactor { factor }

fun decayingEpsilon(factor: Double, decayRate: Double, minFactor: Double    ): ExplorationFactor {
    var epsilon = factor
    return ExplorationFactor {
        epsilon = (epsilon * decayRate).coerceAtLeast(minFactor)
        epsilon
    }
}
