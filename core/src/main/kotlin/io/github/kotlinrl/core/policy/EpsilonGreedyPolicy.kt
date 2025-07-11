package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable
import kotlin.random.*

class EpsilonGreedyPolicy<State, Action>(
    stateActionListProvider: StateActionListProvider<State, Action>,
    qTable: QTable<State, Action>,
    private val explorationFactor: ExplorationFactor,
    private val rng: Random = Random.Default
) : Policy<State, Action> {
    private val randomPolicy = randomPolicy(stateActionListProvider, rng)
    private val greedyPolicy = greedyPolicy(stateActionListProvider, qTable)

    override fun invoke(state: State): Action {
        return if (rng.nextDouble() < explorationFactor()) {
            randomPolicy(state)
        } else {
            greedyPolicy(state)
        }
    }
}