package io.github.kotlinrl.core.policy

import kotlin.random.*

class EpsilonGreedyPolicy<State, Action>(
    stateActionListProvider: StateActionListProvider<State, Action>,
    Q: QFunction<State, Action>,
    private val explorationFactor: ExplorationFactor,
    private val rng: Random = Random.Default
) : Policy<State, Action> {
    private val randomPolicy = randomPolicy(stateActionListProvider, rng)
    private val greedyPolicy = greedyPolicy(stateActionListProvider, Q)

    override fun invoke(state: State): Action {
        return if (rng.nextDouble() < explorationFactor()) {
            randomPolicy(state)
        } else {
            greedyPolicy(state)
        }
    }
}