package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class ConstantAlphaMonteCarloControl<State, Action>(
    private val qTable: QFunction<State, Action>,
    private val gamma: Double = 0.99,
    private val alpha: Double = 0.05,
    private val firstVisitOnly: Boolean = true
) : EpisodeCallback<State, Action> {

    override fun onEpisodeEnd(stats: EpisodeStats<State, Action>) {
        val visited = mutableSetOf<Pair<State, Action>>() // Optional: first-visit only
        var G = 0.0

        for ((s, a, r) in stats.trajectories.asReversed()) {
            G = r + gamma * G
            val key = Pair(s, a)

            if (firstVisitOnly && key in visited) continue

            visited.add(key)
            val oldQ = qTable[s, a]
            qTable[s, a] = oldQ + alpha * (G - oldQ)
        }
    }
}
