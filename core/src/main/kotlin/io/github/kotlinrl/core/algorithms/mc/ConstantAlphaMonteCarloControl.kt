package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class ConstantAlphaMonteCarloControl<State, Action>(
    private val qTable: QFunction<State, Action>,
    private val gamma: Double = 0.99,
    private val alpha: Double = 0.05,
    private val firstVisitOnly: Boolean = true,
    private val stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultKeyFunction
) : TrajectoryLearner<State, Action> {

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        val visited = mutableSetOf<StateActionKey<*, *>>()
        var G = 0.0

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateActionKeyFunction(s, a)

            if (firstVisitOnly && key in visited) continue

            visited.add(key)
            val oldQ = qTable[s, a]
            qTable[s, a] = oldQ + alpha * (G - oldQ)
        }
    }
}
