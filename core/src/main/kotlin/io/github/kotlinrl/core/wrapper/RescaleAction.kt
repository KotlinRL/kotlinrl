package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class RescaleAction<
        O,
        Num : Number,
        D : Dimension,
        OS : Space<O>,
        >(
    env: Env<O, NDArray<Num, D>, OS, Box<Num, D>>,
    private val minAction: NDArray<Num, D>, // typically filled with -1 or 0
    private val maxAction: NDArray<Num, D>  // typically filled with 1
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

    // The agent-facing action space
    override val actionSpace: Box<Num, D> = Box(minAction, maxAction, minAction.dtype)

    override val observationSpace: OS
        get() = env.observationSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> =
        env.reset(seed, options)

    override fun step(action: NDArray<Num, D>): Transition<O> {
        val innerBox = env.actionSpace
        val scaled = rescale(
            x = action,
            srcLow = minAction,
            srcHigh = maxAction,
            tgtLow = innerBox.low,
            tgtHigh = innerBox.high,
            dim = action.dim
        )
        return env.step(scaled)
    }
}
