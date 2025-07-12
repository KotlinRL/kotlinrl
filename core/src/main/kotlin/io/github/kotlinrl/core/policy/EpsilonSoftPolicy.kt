package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable
import kotlin.random.*

class EpsilonSoftPolicy(
    private val stateActionListProvider: StateActionListProvider<IntArray, Int>,
    private val qTable: QTable,
    private val epsilon: ExplorationFactor,
    rng: Random = Random.Default
) : ProbabilisticPolicy<IntArray, Int>(rng) {

    override fun invoke(state: IntArray): Int {
        val availableActions = stateActionListProvider(state)

        val greedy = qTable.bestAction(state)

        val eps = epsilon()
        val uniformProb = eps / availableActions.size
        val greedyProb = 1.0 - eps + uniformProb

        val probabilities = availableActions.map { action ->
            if (action == greedy) greedyProb else uniformProb
        }

        return calculateAndSample(probabilities, availableActions)
    }
}