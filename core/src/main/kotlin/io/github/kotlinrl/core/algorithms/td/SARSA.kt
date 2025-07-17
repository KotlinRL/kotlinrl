package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class SARSA<State, Action>(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {
    private var action: Action? = null

    override fun invoke(trajectory: Trajectory<State, Action>) {
        if (action == null) {
            action = trajectory.action
            return
        }
        val aPrime = action!!
        val (s, a, r, sPrime, terminated, truncated, _) = trajectory
        val done = terminated || truncated

        val currentValue = qTable[s, a]
        val nextValue = if (done) 0.0 else qTable[sPrime, aPrime]

        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        qTable[s, a] = updated
    }
}


