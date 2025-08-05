package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class IncrementalMonteCarloValueFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    private val firstVisitOnly: Boolean = true,
) : TrajectoryValueFunctionEstimator<State, Action> {

    override fun estimate(V: EnumerableValueFunction<State>, trajectory: Trajectory<State, Action>): EnumerableValueFunction<State> {
        val visited = mutableSetOf<Comparable<*>>()
        var G = 0.0
        var newV = V

        for ((s, _, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateKey(s)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val oldV = newV[s]
            val updatedV = oldV + alpha() * (G - oldV)

            newV = newV.update(s, updatedV)
        }

        return newV
    }
}
