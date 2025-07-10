package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

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