package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class ClipAction<
        O,
        Num : Number,
        D : Dimension,
        OS : Space<O>
        >(
    env: Env<O, NDArray<Num, D>, OS, Box<Num, D>>
) : Wrapper<
        O,
        NDArray<Num, D>,
        OS,
        Box<Num, D>,
        O,
        NDArray<Num, D>,
        OS,
        Box<Num, D>
        >(env) {

    override val actionSpace: Box<Num, D>
        get() = env.actionSpace

    override val observationSpace: OS
        get() = env.observationSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> =
        env.reset(seed, options)

    override fun step(action: NDArray<Num, D>): Transition<O> {
        val clipped = clipToBox(action, env.actionSpace)
        return env.step(clipped)
    }
}
