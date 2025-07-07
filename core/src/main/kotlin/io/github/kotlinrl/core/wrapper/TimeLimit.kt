package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class TimeLimit<O, A, OS : Space<O>, AS : Space<A>>(
    env: Env<O, A, OS, AS>,
    private val maxEpisodeSteps: Int
) : SimpleWrapper<O, A, OS, AS>(env) {

    private var elapsedSteps = 0

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> {
        elapsedSteps = 0
        return env.reset(seed, options)
    }

    override fun step(act: A): Transition<O> {
        val transition = env.step(act)
        elapsedSteps += 1

        val reachedTimeLimit = elapsedSteps >= maxEpisodeSteps

        val newTruncated = transition.truncated || (reachedTimeLimit && !transition.terminated)
        val newTerminated = transition.terminated

        val newInfo = transition.info.toMutableMap()
        if (reachedTimeLimit && !transition.terminated) {
            newInfo["TimeLimit.truncated"] = "true"
        }

        return transition.copy(
            terminated = newTerminated,
            truncated = newTruncated,
            info = newInfo
        )
    }
}
