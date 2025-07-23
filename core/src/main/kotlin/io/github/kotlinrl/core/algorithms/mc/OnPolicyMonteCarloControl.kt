package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloControl<State, Action>(
    private val qTable: QFunction<State, Action>,
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true,
    private val stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultKeyFunction
) : TrajectoryLearner<State, Action> {

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

