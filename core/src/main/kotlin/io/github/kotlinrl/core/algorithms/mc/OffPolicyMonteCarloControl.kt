package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OffPolicyMonteCarloControl<State, Action>(
    private val qTable: QFunction<State, Action>,
    private val gamma: Double,
    private val behaviorPolicy: ProbabilisticPolicy<State, Action>,
    private val targetPolicy: MutablePolicy<State, Action>
) : EpisodeCallback<State, Action> {

    private val C: MutableMap<Pair<State, Action>, Double> = mutableMapOf()

    override fun onEpisodeEnd(stats: EpisodeStats<State, Action>) {
        var G = 0.0
        var W = 1.0

        for ((s, a, r) in stats.trajectories.asReversed()) {
            G = gamma * G + r

            val key = Pair(s, a)
            val c = C.getOrDefault(key, 0.0)
            C[key] = c + W

            val q = qTable[s, a]
            qTable[s, a] = q + (W / C[key]!!) * (G - q)

            targetPolicy[s] = qTable.bestAction(s)

            if (a != targetPolicy(s)) break
            W /= behaviorPolicy.probability(s, a)
        }
    }
}