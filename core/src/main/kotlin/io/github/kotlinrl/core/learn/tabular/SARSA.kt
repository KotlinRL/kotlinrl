package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable

class SARSA(
    qTable: QTable,
    alpha: Double,
    gamma: Double
) : TabularTDLearning(qTable, alpha, gamma) {

    override fun invoke(trajectory: Trajectory<IntArray, Int>) {
        val aPrime = action ?: return
        val sPrime = trajectory.nextState
        val a = trajectory.action
        val s = trajectory.state
        val r = trajectory.reward
        val done = trajectory.terminated || trajectory.truncated

        val currentValue = qTable[s, a]
        val nextValue = if (done) 0.0 else qTable[sPrime, aPrime]

        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}


