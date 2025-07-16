package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloControl(
    private val qTable: QTable,
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true
) : EpisodeCallback<IntArray, Int> {

    private val returns: MutableMap<List<Int>, Int> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<IntArray, Int>) {
        val episode = stats.trajectories
        val visited = mutableSetOf<List<Int>>()
        var G = 0.0

        for ((s, a, r) in episode.asReversed()) {
            G = r + gamma * G
            val key = (s + a).toList()

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

