package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class RecordEpisodeStatistics<
        State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>
        >(
    env: Env<State, Action, ObservationSpace, ActionSpace>
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var episodeReward = 0.0
    private var episodeLength = 0

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> {
        episodeReward = 0.0
        episodeLength = 0
        return env.reset(seed, options)
    }

    override fun step(action: Action): Transition<State> {
        val t = env.step(action)
        episodeReward += t.reward
        episodeLength += 1

        val done = t.terminated || t.truncated
        val newInfo = t.info.toMutableMap()
        if (done) {
            // Gymnasium convention: "episode" is a dict with "r" (reward), "l" (length)
            val episodeStats = mapOf(
                "r" to episodeReward.toString(),
                "l" to episodeLength.toString()
            )
            newInfo["episode"] = episodeStats.toString() // as String, or use JSON if desired

            // Reset counters for next episode
            episodeReward = 0.0
            episodeLength = 0
        }
        return t.copy(info = newInfo)
    }
}
