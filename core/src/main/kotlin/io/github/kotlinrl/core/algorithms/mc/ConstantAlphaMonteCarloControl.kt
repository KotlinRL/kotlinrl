package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.algorithms.QTable
import io.github.kotlinrl.core.train.EpisodeCallback
import io.github.kotlinrl.core.train.EpisodeStats

class ConstantAlphaMonteCarloControl(
    private val qTable: QTable,
    private val gamma: Double,
    private val alpha: Double,
    private val firstVisitOnly: Boolean = true
) : EpisodeCallback<IntArray, Int> {

    override fun onEpisodeEnd(stats: EpisodeStats<IntArray, Int>) {
        val episode = stats.trajectories.map { Triple(it.state, it.action, it.reward) }
        val visited = mutableSetOf<List<Int>>() // Optional: first-visit only
        var G = 0.0

        for ((state, action, reward) in episode.asReversed()) {
            G = reward + gamma * G
            val key = (state + action).toList()

            if (firstVisitOnly && key in visited) continue

            visited.add(key)
            val oldQ = qTable[state, action]
            qTable[state, action] = oldQ + alpha * (G - oldQ)
        }
    }
}
