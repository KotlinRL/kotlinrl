package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class NormalizeObservation<
        Num : Number,
        D : Dimension,
        Action,
        OS : Space<NDArray<Num, D>>,
        AS : Space<Action>
        >(
    env: Env<NDArray<Num, D>, Action, OS, AS>,
    private val mean: NDArray<Num, D>,
    private val std: NDArray<Num, D>,
    private val epsilon: Double = 1e-8
) : SimpleWrapper<NDArray<Num, D>, Action, OS, AS>(env) {

    private fun normalize(obs: NDArray<Num, D>): NDArray<Num, D> {
        val obsArr = obs.data
        val meanArr = mean.data
        val stdArr = std.data
        val normed = DoubleArray(obsArr.size) { i ->
            (obsArr[i].toDouble() - meanArr[i].toDouble()) / maxOf(stdArr[i].toDouble(), epsilon)
        }
        @Suppress("UNCHECKED_CAST")
        return mk.ndarray(normed.toList(), obs.shape, obs.dim) as NDArray<Num, D>
    }

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<NDArray<Num, D>> {
        val initial = env.reset(seed, options)
        return InitialState(observation = normalize(initial.observation), info = initial.info)
    }

    override fun step(act: Action): Transition<NDArray<Num, D>> {
        val t = env.step(act)
        return t.copy(observation = normalize(t.observation))
    }
}
