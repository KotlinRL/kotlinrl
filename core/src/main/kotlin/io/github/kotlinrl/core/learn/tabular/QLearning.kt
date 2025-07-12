package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable

class QLearning(
    qTable: QTable,
    alpha: Double,
    gamma: Double
) : TabularTDLearning(qTable, alpha, gamma) {

    override fun invoke(experience: Experience<IntArray, Int>) {
        val s = experience.state
        val a = experience.action
        val sPrime = experience.transition.observation
        val r = experience.transition.reward
        val terminated = experience.transition.terminated

        val currentValue = qTable[s + a]
        val nextValue = if (terminated) 0.0 else qTable.maxValue(sPrime)
        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s + a] = updated
    }
}
