package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.*
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray
import kotlin.math.*
import kotlin.random.*

class SoftmaxPolicy(
    private val stateActionListProvider: StateActionListProvider<IntArray, Int>,
    private val qTable: QTable,
    private val temperature: ExplorationFactor,
    rng: Random = Random.Default
) : ProbabilisticPolicy<IntArray, Int>(rng) {

    override fun invoke(state: IntArray): Int {
        val availableActions = stateActionListProvider(state)
        val T = temperature()
        val qValues = qTable.qValues(state)
        val maxQ = qTable.maxValue(state)
        val scaled = qValues.map { exp((it - maxQ) / T) }.toDoubleArray().toList()

        return calculateAndSample(scaled, availableActions)
    }
}