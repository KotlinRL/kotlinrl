package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.Env
import io.github.kotlinrl.core.env.InitialState
import io.github.kotlinrl.core.env.Transition
import io.github.kotlinrl.core.space.Space

open class SimpleWrapper<O, A, OS : Space<O>, AS : Space<A>>(
    env: Env<O, A, OS, AS>
) : Wrapper<O, A, OS, AS, O, A, OS, AS>(env) {

    override fun step(act: A): Transition<O> = env.step(act)

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O>  = env.reset(seed, options)

    override val observationSpace: OS
        get() = env.observationSpace

    override val actionSpace: AS
        get() = env.actionSpace
}