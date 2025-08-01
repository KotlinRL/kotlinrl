package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import java.io.*

class Monitor<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    logPath: String = "monitor.csv"
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var episodeReward = 0.0
    private var episodeLength = 0
    private val logFile = File(logPath).apply {
        if (!exists()) writeText("episode,reward,length\n")
    }

    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        if (episodeLength > 0) {
            // Log to file beforeStep resetting counters (if previous episode finished)
            logFile.appendText("${System.currentTimeMillis()},$episodeReward,$episodeLength\n")
        }
        episodeReward = 0.0
        episodeLength = 0
        return env.reset(seed, options)
    }

    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        episodeReward += t.reward
        episodeLength += 1
        if (t.terminated || t.truncated) {
            // Log to file at end of episode
            logFile.appendText("${System.currentTimeMillis()},$episodeReward,$episodeLength\n")
            // Optionally, also attach stats to info:
            val newInfo = t.info.toMutableMap()
            newInfo["monitor"] = "reward=$episodeReward,length=$episodeLength"
            return t.copy(info = newInfo)
        }
        return t
    }
}
