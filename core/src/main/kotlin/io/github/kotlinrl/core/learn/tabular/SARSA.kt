package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable

class SARSA<State, Action>(
    qTable: QTable<State, Action>,
    alpha: Double,
    gamma: Double
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {

    override fun invoke(experience: Experience<State, Action>) {
        val a = experience.priorAction ?: return
        val aPrime = action ?: error("Next action not yet recorded")
        val s = experience.priorState
        val sPrime = experience.transition.observation
        val r = experience.transition.reward

        val currentValue = qTable[s, a]
        val nextValue = if (experience.transition.terminated) 0.0 else qTable[sPrime, aPrime]

        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}


