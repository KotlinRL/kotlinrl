package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable

class QLearning(
    qTable: QTable,
    alpha: Double,
    gamma: Double
) : TabularTDLearning(qTable, alpha, gamma) {

    override fun invoke(trajectory: Trajectory<IntArray, Int>) {
        val s = trajectory.state
        val a = trajectory.action
        val sPrime = trajectory.nextState
        val r = trajectory.reward
        val done = trajectory.terminated || trajectory.truncated

        val currentValue = qTable[s, a]
        val nextValue = if (done) 0.0 else qTable.maxValue(sPrime)
        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}
