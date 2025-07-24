package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class NormalizeReward<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    private val epsilon: Double = 1e-8
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private val stats = RunningStats()

    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> =
        env.reset(seed, options)

    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        stats.update(t.reward)
        val normalized = (t.reward - stats.mean) / maxOf(stats.std, epsilon)
        return t.copy(reward = normalized)
    }
}