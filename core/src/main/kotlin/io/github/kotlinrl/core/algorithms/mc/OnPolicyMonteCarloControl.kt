package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloControl<State, Action>(
    qTable: QFunction<State, Action>,
    gamma: Double,
    private val firstVisitOnly: Boolean = true,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultKeyFunction
) : MCLearning<State, Action>(qTable, gamma, stateActionKeyFunction) {

    private val returns: MutableMap<StateActionKey<*, *>, Int> = mutableMapOf()

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        val visited = mutableSetOf<StateActionKey<*, *>>()
        var G = 0.0

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateActionKeyFunction(s, a)

            if (firstVisitOnly && key in visited) continue

            visited.add(key)
            val count = returns.getOrDefault(key, 0)
            val oldQ = qTable[s, a]
            val newQ = oldQ + (G - oldQ) / (count + 1)
            qTable[s, a] = newQ
            returns[key] = count + 1
        }
    }
}

