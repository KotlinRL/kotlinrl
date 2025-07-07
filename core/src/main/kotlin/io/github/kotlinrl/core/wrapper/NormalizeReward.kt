package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class NormalizeReward<
        O, A, OS : Space<O>, AS : Space<A>
        >(
    env: Env<O, A, OS, AS>,
    private val epsilon: Double = 1e-8
) : SimpleWrapper<O, A, OS, AS>(env) {

    private val stats = RunningStats()

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> =
        env.reset(seed, options)

    override fun step(act: A): Transition<O> {
        val t = env.step(act)
        stats.update(t.reward)
        val normalized = (t.reward - stats.mean) / maxOf(stats.std, epsilon)
        return t.copy(reward = normalized)
    }
}