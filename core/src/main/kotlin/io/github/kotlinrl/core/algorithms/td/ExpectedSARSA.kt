package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.QFunction
import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithms.QTable
import io.github.kotlinrl.core.policy.*

class ExpectedSARSA<State, Action>(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val policyProbabilities: PolicyProbabilities<State, Action>
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {

    override fun invoke(trajectory: Trajectory<State, Action>) {
        val (s, a, r, sPrime, terminated, truncated, _) = trajectory
        val done = terminated || truncated
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
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}
