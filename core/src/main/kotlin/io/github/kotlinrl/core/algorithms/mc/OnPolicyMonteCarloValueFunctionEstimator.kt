package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloValueFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true,
) : TrajectoryValueFunctionEstimator<State, Action> {

    private val returnsCount: MutableMap<Comparable<*>, Int> = mutableMapOf()
    private val returnsSum: MutableMap<Comparable<*>, Double> = mutableMapOf()

    override fun estimate(V: EnumerableValueFunction<State>, trajectory: Trajectory<State, Action>): EnumerableValueFunction<State> {
        val visited = mutableSetOf<Comparable<*>>()
        var G = 0.0
        var updatedV = V

        for ((s, _, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateKey(s)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val count = returnsCount.getOrDefault(key, 0) + 1
            val sum = returnsSum.getOrDefault(key, 0.0) + G

            returnsCount[key] = count
            returnsSum[key] = sum

            val averageReturn = sum / count
            updatedV = updatedV.update(s, averageReturn)
        }

        return updatedV
    }
}
