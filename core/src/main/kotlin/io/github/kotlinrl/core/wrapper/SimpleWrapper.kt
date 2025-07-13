package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

open class SimpleWrapper<State, Action, StateSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, StateSpace, ActionSpace>
) : Wrapper<State, Action, StateSpace, ActionSpace, State, Action, StateSpace, ActionSpace>(env) {

    override fun step(action: Action): Transition<State> = env.step(action)

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State>  = env.reset(seed, options)

    override val observationSpace: StateSpace
        get() = env.observationSpace

    override val actionSpace: ActionSpace
        get() = env.actionSpace
}