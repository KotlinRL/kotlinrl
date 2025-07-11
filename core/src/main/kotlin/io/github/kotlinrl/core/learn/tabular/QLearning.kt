package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable

class QLearning<State, Action>(
    qTable: QTable<State, Action>,
    alpha: Double,
    gamma: Double
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {

    override fun invoke(experience: Experience<State, Action>) {
        val s = experience.priorState
        val a = experience.priorAction ?: return
        val sPrime = experience.transition.observation
        val r = experience.transition.reward
        val terminated = experience.transition.terminated

        val currentValue = qTable[s, a]
        val nextValue = if (terminated) 0.0 else qTable.maxValue(sPrime)
        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}
