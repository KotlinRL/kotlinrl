package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class ConstantAlphaMonteCarloControl(
    private val qTable: QTable,
    private val gamma: Double = 0.99,
    private val alpha: Double = 0.05,
    private val firstVisitOnly: Boolean = true
) : EpisodeCallback<IntArray, Int> {

    override fun onEpisodeEnd(stats: EpisodeStats<IntArray, Int>) {
        val episode = stats.trajectories
        val visited = mutableSetOf<List<Int>>() // Optional: first-visit only
        var G = 0.0

        for ((s, a, r) in episode.asReversed()) {
            G = r + gamma * G
            val key = (s + a).toList()

            if (firstVisitOnly && key in visited) continue

            visited.add(key)
            val oldQ = qTable[s, a]
            qTable[s, a] = oldQ + alpha * (G - oldQ)
        }
    }
}
