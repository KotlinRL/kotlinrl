package io.github.kotlinrl.integration.gymnasium

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import io.github.kotlinrl.open.env.*
import open.rl.env.EnvOuterClass.DType.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import kotlin.random.*

@Suppress("UNCHECKED_CAST")
internal class RemoteEnvClient<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    envName: String,
    seed: Int? = null,
    render: Boolean = true,
    options: Map<String, Any?> = emptyMap(),
    host: String = "localhost:50051"

) : Env<State, Action, ObservationSpace, ActionSpace> {
    internal val env = RemoteEnv(envName, render, options.toStruct(), host)
    override val random: Random = seed?.let { Random(it) } ?: Random.Default
    override val metadata: Map<String, Any?> = env.metadata.toMap()
    override val observationSpace = env.observationSpace.toTypedSpace(seed) as ObservationSpace
    override val actionSpace = env.actionSpace.toTypedSpace(seed) as ActionSpace

    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        val (state, info) = env.reset(seed, options?.toStruct())
        return InitialState(
            info = info.toMap(),
            state = state.toTypedState()
        )
    }

    override fun step(action: Action): StepResult<State> {
        val (state, reward, terminated, truncated, info) = env.step(when(action) {
            is String -> action(action as String)
            is Int -> action(action as Int)
            is Double -> action(action as Double)
            is NDArray<*, *> -> action(
                dtype = when(action.dtype) {
                    ByteDataType -> uint8
                    IntDataType -> int32
                    LongDataType -> int64
                    FloatDataType -> float32
                    DoubleDataType -> float64
                   else -> error("Invalid dtype: ${action.dtype}")
                },
                shape = action.shape,
                data = when(action.dtype) {
                    ByteDataType -> action.data.getByteArray()
                    IntDataType -> action.data.getIntArray().toByteArray()
                    LongDataType -> action.data.getLongArray().toByteArray()
                    FloatDataType -> action.data.getFloatArray().toByteArray()
                    DoubleDataType -> action.data.getDoubleArray().toByteArray()
                    else -> error("Invalid dtype: ${action.dtype}")
                }
            )
            else -> TODO()
        })
        return StepResult(
            state = state.toTypedState() as State,
            reward = reward,
            terminated = terminated,
            truncated = truncated,
            info = info.toMap()
        )
    }

    override fun render(): Rendering = env.render().toRendering()

    override fun close() {
        env.close()
    }
}
