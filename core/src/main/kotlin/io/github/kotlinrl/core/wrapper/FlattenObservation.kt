package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*

class FlattenObservation<
        Num : Number,
        WrappedObservation,
        Action,
        WrappedObservationSpace : Space<WrappedObservation>,
        ActionSpace : Space<Action>
        >(
    env: Env<WrappedObservation, Action, WrappedObservationSpace, ActionSpace>,
    private val dtype: DataType
) : Wrapper<
        NDArray<Num, D1>,
        Action,
        Box<Num, D1>,
        ActionSpace,
        WrappedObservation,
        Action,
        WrappedObservationSpace,
        ActionSpace
        >(env) {

    override val observationSpace: Box<Num, D1> by lazy {
        val sampleObs = env.observationSpace.sample()
        val flatList = flattenObservation(sampleObs, dtype)
        val flatLength = flatList.size

        @Suppress("UNCHECKED_CAST")
        val low = when (dtype) {
            DoubleDataType -> mk.ndarray<Double, D1>(List(flatLength) { Double.NEGATIVE_INFINITY }, intArrayOf(flatLength)) as NDArray<Num, D1>
            FloatDataType -> mk.ndarray<Float, D1>(List(flatLength) { Float.NEGATIVE_INFINITY }, intArrayOf(flatLength)) as NDArray<Num, D1>
            IntDataType -> mk.ndarray<Int, D1>(List(flatLength) { Int.MIN_VALUE }, intArrayOf(flatLength)) as NDArray<Num, D1>
            LongDataType -> mk.ndarray<Long, D1>(List(flatLength) { Long.MIN_VALUE }, intArrayOf(flatLength)) as NDArray<Num, D1>
            else -> throw IllegalArgumentException("Unsupported dtype: $dtype")
        }
        @Suppress("UNCHECKED_CAST")
        val high = when (dtype) {
            DoubleDataType -> mk.ndarray<Double, D1>(List(flatLength) { Double.POSITIVE_INFINITY }, intArrayOf(flatLength)) as NDArray<Num, D1>
            FloatDataType -> mk.ndarray<Float, D1>(List(flatLength) { Float.POSITIVE_INFINITY }, intArrayOf(flatLength)) as NDArray<Num, D1>
            IntDataType -> mk.ndarray<Int, D1>(List(flatLength) { Int.MAX_VALUE }, intArrayOf(flatLength)) as NDArray<Num, D1>
            LongDataType -> mk.ndarray<Long, D1>(List(flatLength) { Long.MAX_VALUE }, intArrayOf(flatLength)) as NDArray<Num, D1>
            else -> throw IllegalArgumentException("Unsupported dtype: $dtype")
        }
        Box(low, high, dtype)
    }

    override val actionSpace: ActionSpace
        get() = env.actionSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<NDArray<Num, D1>> {
        val initial = env.reset(seed, options)
        val flat = toNDArray<Num>(flattenObservation(initial.observation, dtype), dtype)
        return InitialState(
            observation = flat,
            info = initial.info
        )
    }

    override fun step(act: Action): Transition<NDArray<Num, D1>> {
        val t = env.step(act)
        val flat = toNDArray<Num>(flattenObservation(t.observation, dtype), dtype)
        return Transition(
            observation = flat,
            reward = t.reward,
            terminated = t.terminated,
            truncated = t.truncated,
            info = t.info
        )
    }
}
