package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.TrajectoryQFunctionEstimator

class IncrementalMonteCarloQFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    private val firstVisitOnly: Boolean = true,
) : TrajectoryQFunctionEstimator<State, Action> {

    override fun estimate(
        q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        var currentQ = q
        val visited = mutableSetOf<StateActionKey<*, *>>()
        var G = 0.0

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateActionKey(s, a)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val oldQ = currentQ[s, a]
            val updatedQ = oldQ + alpha() * (G - oldQ)
            currentQ = currentQ.update(s, a, updatedQ)
        }

        return currentQ
    }
}
