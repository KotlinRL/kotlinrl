package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable

class SARSA(
    qTable: QTable,
    alpha: Double,
    gamma: Double
) : TabularTDLearning(qTable, alpha, gamma) {

    override fun invoke(experience: Experience<IntArray, Int>) {
        val aPrime = action ?: return
        val sPrime = experience.transition.observation
        val a = experience.action
        val s = experience.state
        val r = experience.transition.reward

        val currentValue = qTable[s + a]
        val nextValue = if (experience.transition.terminated) 0.0 else qTable[sPrime + aPrime]

        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s + a] = updated
    }
}


