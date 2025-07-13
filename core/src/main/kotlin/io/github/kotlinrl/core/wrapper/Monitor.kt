package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import java.io.*

class Monitor<
        O, A, OS : Space<O>, AS : Space<A>
        >(
    env: Env<O, A, OS, AS>,
    logPath: String = "monitor.csv"
) : SimpleWrapper<O, A, OS, AS>(env) {

    private var episodeReward = 0.0
    private var episodeLength = 0
    private val logFile = File(logPath).apply {
        if (!exists()) writeText("episode,reward,length\n")
    }

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> {
        if (episodeLength > 0) {
            // Log to file before resetting counters (if previous episode finished)
            logFile.appendText("${System.currentTimeMillis()},$episodeReward,$episodeLength\n")
        }
        episodeReward = 0.0
        episodeLength = 0
        return env.reset(seed, options)
    }

    override fun step(action: A): Transition<O> {
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
