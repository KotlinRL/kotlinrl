package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OffPolicyMonteCarloControl(
    private val qTable: QTable,
    private val gamma: Double,
    private val behaviorPolicy: ProbabilisticPolicy<IntArray, Int>,
    private val targetPolicy: MutablePolicy<IntArray, Int>
) : EpisodeCallback<IntArray, Int> {

    private val C: MutableMap<List<Int>, Double> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<IntArray, Int>) {
        val episode = stats.trajectories
        var G = 0.0
        var W = 1.0

        for (trajectory in episode.asReversed()) {
            val s = trajectory.state
            val a = trajectory.action
            val r = trajectory.reward
            G = gamma * G + r

            val key = (s + a).toList()
            val c = C.getOrDefault(key, 0.0)
            C[key] = c + W

            val q = qTable[s, a]
            qTable[s, a] = q + (W / C[key]!!) * (G - q)

            targetPolicy[s] = qTable.bestAction(s)

            if (a != targetPolicy[s]) break
            W /= behaviorPolicy.probability(s, a)
        }
    }
}