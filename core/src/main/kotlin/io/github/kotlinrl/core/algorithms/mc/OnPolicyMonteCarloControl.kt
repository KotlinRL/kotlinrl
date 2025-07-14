package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloControl(
    private val qTable: QTable,
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true
) : EpisodeCallback<IntArray, Int> {

    private val returns: MutableMap<List<Int>, Int> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<IntArray, Int>) {
        val episode = stats.trajectories.map { Triple(it.state, it.action, it.reward) }
        val visited = mutableSetOf<List<Int>>()
        var G = 0.0

        for ((state, action, reward) in episode.asReversed()) {
            G = reward + gamma * G
            val key = (state + action).toList()

            if (firstVisitOnly && key in visited) continue

            visited.add(key)
            val count = returns.getOrDefault(key, 0)
            val oldQ = qTable[state, action]
            val newQ = oldQ + (G - oldQ) / (count + 1)
            qTable[state, action] = newQ
            returns[key] = count + 1
        }

    }
}

