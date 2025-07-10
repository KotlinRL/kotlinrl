package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.MutableQFunction
import io.github.kotlinrl.core.policy.*

class ExpectedSARSA<State, Action>(
    Q: MutableQFunction<State, Action>,
    alpha: Double,
    gamma: Double,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val policyProbabilities: PolicyProbabilities<State, Action>
) : TabularTDLearning<State, Action>(Q, alpha, gamma) {

    override fun invoke(experience: Experience<State, Action>) {
        val a = experience.priorAction ?: return
        val s = experience.priorState
        val sPrime = experience.transition.observation
        val r = experience.transition.reward

        val currentValue = Q(s, a)

        val expectedValue = if (experience.transition.terminated) {
            0.0
        } else {
            val probs = policyProbabilities(sPrime)
            val actions = stateActionListProvider(sPrime)
            actions.sumOf { aPrime ->
                probs.getOrDefault(aPrime, 0.0) * Q(sPrime, aPrime)
            }
        }

        val target = r + gamma * expectedValue
        val updated = currentValue + alpha * (target - currentValue)

        Q.update(s, a, updated)
    }
}
