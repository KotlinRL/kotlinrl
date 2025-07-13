package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable
import io.github.kotlinrl.core.policy.*

class ExpectedSARSA(
    qTable: QTable,
    alpha: Double,
    gamma: Double,
    private val stateActionListProvider: StateActionListProvider<IntArray, Int>,
    private val policyProbabilities: PolicyProbabilities<IntArray, Int>
) : TabularTDLearning(qTable, alpha, gamma) {

    override fun invoke(trajectory: Trajectory<IntArray, Int>) {
        val a = trajectory.action
        val s = trajectory.state
        val sPrime = trajectory.nextState
        val r = trajectory.reward
        val done = trajectory.terminated || trajectory.truncated
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
