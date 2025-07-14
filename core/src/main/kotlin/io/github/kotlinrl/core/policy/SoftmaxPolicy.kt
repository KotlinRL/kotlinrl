package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.ExplorationFactor
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*
import kotlin.random.*

class SoftmaxPolicy(
    private val qTable: QTable,
    private val temperature: ExplorationFactor,
    rng: Random
) : ProbabilisticPolicy<IntArray, Int>(rng) {

    override fun actionScores(state: IntArray): List<Pair<Int, Double>> {
        val temperature = temperature()
        val qValues = qTable.qValues(state)
        val mapIndexed = qValues.mapIndexed { action, q -> action to exp(q / temperature) }
        return mapIndexed.toList()
    }
}