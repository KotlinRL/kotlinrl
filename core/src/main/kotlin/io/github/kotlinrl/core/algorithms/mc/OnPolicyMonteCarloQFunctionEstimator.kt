package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloQFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true,
) : MonteCarloQFunctionEstimator<State, Action> {
    private val returns: MutableMap<StateActionKey<*, *>, Int> = mutableMapOf()

    override fun estimate(
        q: QFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): QFunction<State, Action> {
        val visited = mutableSetOf<StateActionKey<*, *>>()
        var G = 0.0
        var currentQ = q

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateActionKey(s, a)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val count = returns.getOrDefault(key, 0)
            val oldQ = currentQ[s, a]
            val newQ = oldQ + (G - oldQ) / (count + 1)

            returns[key] = count + 1
            currentQ = currentQ.update(s, a, newQ)
        }

        return currentQ
    }
}
