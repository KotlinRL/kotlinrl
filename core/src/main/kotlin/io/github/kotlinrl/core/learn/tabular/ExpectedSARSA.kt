package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable
import io.github.kotlinrl.core.policy.*

class ExpectedSARSA<State, Action>(
    qTable: QTable<State, Action>,
    alpha: Double,
    gamma: Double,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val policyProbabilities: PolicyProbabilities<State, Action>
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {

    override fun invoke(experience: Experience<State, Action>) {
        val a = experience.priorAction ?: return
        val s = experience.priorState
        val sPrime = experience.transition.observation
        val r = experience.transition.reward

        val currentValue = qTable[s, a]

        val expectedValue = if (experience.transition.terminated) {
            0.0
        } else {
            val probs = policyProbabilities(sPrime)
            val actions = stateActionListProvider(sPrime)
            actions.sumOf { aPrime ->
                probs.getOrDefault(aPrime, 0.0) * qTable[sPrime, aPrime]
            }
        }

        val target = r + gamma * expectedValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}
