package io.github.kotlinrl.core.learn.mcc

import io.github.kotlinrl.core.learn.QTable
import io.github.kotlinrl.core.train.EpisodeCallback
import io.github.kotlinrl.core.train.EpisodeStats

class MonteCarloControl(
    private val qTable: QTable,
    private val gamma: Double
) : EpisodeCallback<IntArray, Int> {

    private val returns: MutableMap<IntArray, Int> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<IntArray, Int>) {
        val episode = stats.experiences.map { Triple(it.state, it.action, it.transition.reward) }
        val visited = mutableSetOf<IntArray>()
        var G = 0.0

        for ((state, action, reward) in episode.asReversed()) {
            G = reward + gamma * G
            val key = state + action
            if (visited.none { it contentEquals key }) {
                visited += key
                val count = returns.getOrDefault(key, 0)
                val oldQ = qTable[key]
                val newQ = oldQ + (G - oldQ) / (count + 1)
                qTable[key] = newQ
                returns[key] = count + 1
            }
        }
    }
}

