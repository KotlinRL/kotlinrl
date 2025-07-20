package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloControl<State, Action>(
    private val qTable: QFunction<State, Action>,
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true
) : EpisodeCallback<State, Action> {

    private val returns: MutableMap<Pair<State, Action>, Int> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<State, Action>) {
        val visited = mutableSetOf<Pair<State, Action>>()
        var G = 0.0

        for ((s, a, r) in stats.transitions.asReversed()) {
            G = r + gamma * G
            val key = Pair(s, a)

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

