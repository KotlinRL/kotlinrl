package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class OrderEnforcing<
        O, A, OS : Space<O>, AS : Space<A>
        >(
    env: Env<O, A, OS, AS>
) : SimpleWrapper<O, A, OS, AS>(env) {

    private var needsReset = true

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> {
        needsReset = false
        return env.reset(seed, options)
    }

    override fun step(action: A): Transition<O> {
        if (needsReset) {
            throw IllegalStateException(
                "step() called before reset(), or after episode done. " +
                        "You must call reset() before step()."
            )
        }
        val t = env.step(action)
        if (t.terminated || t.truncated) {
            needsReset = true
        }
        return t
    }
}
