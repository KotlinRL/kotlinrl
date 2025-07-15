package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class SARSA(
    qTable: QTable,
    alpha: Double,
    gamma: Double
) : TabularTDLearning(qTable, alpha, gamma), StepCallback<IntArray, Int> {
    private var action: Int? = null

    override fun afterStep(state: IntArray, action: Int) {
        this.action = action
    }

    override fun invoke(trajectory: Trajectory<IntArray, Int>) {
        val aPrime = action ?: return
        val (s, sPrime, a, r, terminated, truncated, _) = trajectory
        val done = terminated || truncated

        val currentValue = qTable[s, a]
        val nextValue = if (done) 0.0 else qTable[sPrime, aPrime]

        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}


