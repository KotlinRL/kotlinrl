package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class TimeLimit<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    private val maxEpisodeSteps: Int
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var elapsedSteps = 0

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> {
        elapsedSteps = 0
        return env.reset(seed, options)
    }

    override fun step(action: Action): Transition<State> {
        val transition = env.step(action)
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
