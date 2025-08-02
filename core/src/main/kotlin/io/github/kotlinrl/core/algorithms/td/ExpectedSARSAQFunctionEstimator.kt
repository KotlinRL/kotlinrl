package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.policy.StateActionListProvider

class ExpectedSARSAQFunctionEstimator<State, Action>(
    private val alpha: Double,
    private val gamma: Double,
    initialPolicyProbabilities: PolicyProbabilities<State, Action>,
    private val stateActionListProvider: StateActionListProvider<State, Action>
) : TDQFunctionEstimator<State, Action> {
    var policyProbabilities: PolicyProbabilities<State, Action> = initialPolicyProbabilities

    override fun estimate(q: QFunction<State, Action>, transition: Transition<State, Action>): QFunction<State, Action> {
        val (s, a, r, sPrime) = transition
        val done = transition.done
        val currentValue = q[s, a]

        val expectedValue = if (done) {
            0.0
        } else {
            val probs = policyProbabilities(sPrime)
            val actions = stateActionListProvider(sPrime)
            actions.sumOf { aPrime ->
                probs.getOrDefault(aPrime, 0.0) * q[sPrime, aPrime]
            }
        }

        val target = r + gamma * expectedValue
        val updated = currentValue + alpha * (target - currentValue)
        return q.update(s, a, updated)
    }
}