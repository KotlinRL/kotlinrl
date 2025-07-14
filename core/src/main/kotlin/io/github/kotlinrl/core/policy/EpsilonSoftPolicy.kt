package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.ExplorationFactor
import io.github.kotlinrl.core.algorithms.QTable
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import kotlin.random.*

class EpsilonSoftPolicy(
    private val qTable: QTable,
    private val epsilon: ExplorationFactor,
    private val stateActionListProvider: StateActionListProvider<IntArray, Int>,
    rng: Random
) : ProbabilisticPolicy<IntArray, Int>(rng) {

    override fun actionScores(state: IntArray): List<Pair<Int, Double>> {
        val actions = stateActionListProvider(state)
        val greedyAction = qTable.bestAction(state)
        val n = actions.size
        val epsilon = epsilon()

        return actions.map { action ->
            val prob = if (action == greedyAction) {
                (1 - epsilon) + (epsilon / n)
            } else {
                epsilon / n
            }
            action to prob
        }
    }
}