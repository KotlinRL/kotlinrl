package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.QFunction
import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithms.QTable

class QLearning<State, Action>(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {

    override fun invoke(trajectory: Trajectory<State, Action>) {
        val (s, a, r, sPrime, terminated, truncated, _) = trajectory
        val done = terminated || truncated

        val currentValue = qTable[s, a]
        val nextValue = if (done) 0.0 else qTable.maxValue(sPrime)
        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}
