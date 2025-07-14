package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

open class SimpleWrapper<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>
) : Wrapper<State, Action, ObservationSpace, ActionSpace, State, Action, ObservationSpace, ActionSpace>(env) {

    override fun step(action: Action): Transition<State> = env.step(action)

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State>  = env.reset(seed, options)

    override val observationSpace: ObservationSpace
        get() = env.observationSpace

    override val actionSpace: ActionSpace
        get() = env.actionSpace
}