package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable
import kotlin.random.*

class EpsilonGreedyPolicy(
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    qTable: QTable,
    private val explorationFactor: ExplorationFactor,
    private val rng: Random = Random.Default
) : Policy<IntArray, Int> {
    private val randomPolicy = randomPolicy(stateActionListProvider, rng)
    private val greedyPolicy = greedyPolicy(qTable)

    override fun invoke(state: IntArray): Int {
        return if (rng.nextDouble() < explorationFactor()) {
            randomPolicy(state)
        } else {
            greedyPolicy(state)
        }
    }
}