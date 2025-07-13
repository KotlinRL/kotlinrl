package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class ClipAction<
        State,
        Num : Number,
        D : Dimension,
        StateSpace
        : Space<State>
        >(
    env: Env<State, NDArray<Num, D>, StateSpace, Box<Num, D>>
) : Wrapper<
        State,
        NDArray<Num, D>,
        StateSpace,
        Box<Num, D>,
        State,
        NDArray<Num, D>,
        StateSpace,
        Box<Num, D>
        >(env) {

    override val actionSpace: Box<Num, D>
        get() = env.actionSpace

    override val observationSpace: StateSpace
        get() = env.observationSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> =
        env.reset(seed, options)

    override fun step(action: NDArray<Num, D>): Transition<State> {
        val clipped = clipToBox(action, env.actionSpace)
        return env.step(clipped)
    }
}
