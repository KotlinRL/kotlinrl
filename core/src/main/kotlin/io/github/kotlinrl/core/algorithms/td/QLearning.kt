package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithms.QTable

class QLearning(
    qTable: QTable,
    alpha: Double,
    gamma: Double
) : TabularTDLearning(qTable, alpha, gamma) {

    override fun invoke(trajectory: Trajectory<IntArray, Int>) {
        val (s, sPrime, a, r, terminated, truncated, _) = trajectory
        val done = terminated || truncated

        val currentValue = qTable[s, a]
        val nextValue = if (done) 0.0 else qTable.maxValue(sPrime)
        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}
