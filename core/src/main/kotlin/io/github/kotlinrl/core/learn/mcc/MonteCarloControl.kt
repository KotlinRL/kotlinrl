package io.github.kotlinrl.core.learn.mcc

import io.github.kotlinrl.core.learn.QTable
import io.github.kotlinrl.core.train.EpisodeCallback
import io.github.kotlinrl.core.train.EpisodeStats

class MonteCarloControl(
    private val qTable: QTable,
    private val gamma: Double
) : EpisodeCallback<IntArray, Int> {

    private val returns: MutableMap<Pair<IntArray, Int>, MutableList<Double>> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<IntArray, Int>) {
        val episode = stats.experiences.map {
            Triple(it.state, it.action, it.transition.reward)
        }

        val visited = mutableSetOf<Pair<IntArray, Int>>()
        var G = 0.0

        for ((state, action, reward) in episode.asReversed()) {
            G = reward + gamma * G
            val key = state.copyOf() to action
            if (key !in visited) {
                visited += key
                returns.getOrPut(key) { mutableListOf() }.add(G)
                qTable[state + action] = returns[key]!!.average()
            }
        }
    }
}

