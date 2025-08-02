package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.defaultStateKeyFunction

class OnPolicyMonteCarloValueFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true,
    private val stateKeyFunction: StateKeyFunction<State> = ::defaultStateKeyFunction
) : MonteCarloValueFunctionEstimator<State, Action> {

    private val returnsCount: MutableMap<Comparable<*>, Int> = mutableMapOf()
    private val returnsSum: MutableMap<Comparable<*>, Double> = mutableMapOf()

    override fun estimate(v: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State> {
        val visited = mutableSetOf<Comparable<*>>()
        var G = 0.0
        var newV = v

        for ((s, _, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateKeyFunction(s)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val count = returnsCount.getOrDefault(key, 0) + 1
            val sum = returnsSum.getOrDefault(key, 0.0) + G

            returnsCount[key] = count
            returnsSum[key] = sum

            val averageReturn = sum / count
            newV = newV.update(s, averageReturn)
        }

        return newV
    }
}
