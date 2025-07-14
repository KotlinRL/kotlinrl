package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class OrderEnforcing<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var needsReset = true

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> {
        needsReset = false
        return env.reset(seed, options)
    }

    override fun step(action: Action): Transition<State> {
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
