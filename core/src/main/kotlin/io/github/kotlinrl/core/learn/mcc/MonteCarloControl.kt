package io.github.kotlinrl.core.learn.mcc

import io.github.kotlinrl.core.learn.MutableQFunction
import io.github.kotlinrl.core.train.EpisodeCallback
import io.github.kotlinrl.core.train.EpisodeStats

class MonteCarloControl<State, Action>(
    private val Q: MutableQFunction<State, Action>,
    private val gamma: Double
) : EpisodeCallback<State, Action> {

    private val returns: MutableMap<Pair<State, Action>, MutableList<Double>> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<State, Action>) {
        val episode = stats.transitions.zip(stats.actions) { transition, action ->
            Triple(transition.observation, action, transition.reward)
        }

        var G = 0.0
        for ((state, action, reward) in episode.asReversed()) {
            G = reward + gamma * G
            val key = state to action
            returns.getOrPut(key) { mutableListOf() }.add(G)
            Q.update(state, action, returns[key]!!.average())
        }
    }
}

