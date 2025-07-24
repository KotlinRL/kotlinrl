package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.policy.PolicyProbabilities
import io.github.kotlinrl.core.policy.StateActionListProvider

class ExpectedSARSA<State, Action>(
    qTable: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val policyProbabilities: PolicyProbabilities<State, Action>
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {

    override fun invoke(transition: Transition<State, Action>) {
        val (s, a, r, sPrime) = transition
        val done = transition.done
        val currentValue = qTable[s, a]

        val expectedValue = if (done) {
            0.0
        } else {
            val probs = policyProbabilities(sPrime)
            val actions = stateActionListProvider(sPrime)
            actions.sumOf { aPrime ->
                probs.getOrDefault(aPrime, 0.0) * qTable[sPrime, aPrime]
            }
        }

        val target = r + gamma * expectedValue
        val updated = currentValue + alpha() * (target - currentValue)

        qTable[s, a] = updated
    }
}
