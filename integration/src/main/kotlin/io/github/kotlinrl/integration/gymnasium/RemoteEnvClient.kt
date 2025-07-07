package io.github.kotlinrl.integration.gymnasium

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import io.github.kotlinrl.open.env.*
import open.rl.env.EnvOuterClass.DType.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import kotlin.random.*

@Suppress("UNCHECKED_CAST")
internal class RemoteEnvClient<Observation, Action, ObservationSpace : Space<Observation>, ActionSpace : Space<Action>>(
    envName: String,
    seed: Int? = null,
    render: Boolean = true,
    options: Map<String, String> = emptyMap(),
    host: String = "localhost:50051"

) : Env<Observation, Action, ObservationSpace, ActionSpace> {
    internal val env = RemoteEnv(envName, render, options, host)
    override val random: Random = seed?.let { Random(it) } ?: Random.Default
    override val metadata: Map<String, Any> = env.metadata.toMap()
    override val observationSpace = env.observationSpace.toTypedSpace(seed) as ObservationSpace
    override val actionSpace = env.actionSpace.toTypedSpace(seed) as ActionSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<Observation> {
        val (observation, info) = env.reset(seed, options)
        return InitialState(
            info = info.dataMap,
            observation = observation.toTypedObservation()
        )
    }

    override fun step(act: Action): Transition<Observation> {
        val (observation, reward, terminated, truncated, info) = env.step(when(act) {
            is String -> action(act as String)
            is Int -> action(act as Int)
            is Float -> action(act as Float)
            is NDArray<*, *> -> action(
                dtype = when(act.dtype) {
                    ByteDataType -> uint8
                    IntDataType -> int32
                    LongDataType -> int64
                    FloatDataType -> float32
                    DoubleDataType -> float64
                   else -> error("Invalid dtype: ${act.dtype}")
                },
                shape = act.shape,
                data = when(act.dtype) {
                    ByteDataType -> act.data.getByteArray()
                    IntDataType -> act.data.getIntArray().toByteArray()
                    LongDataType -> act.data.getLongArray().toByteArray()
                    FloatDataType -> act.data.getFloatArray().toByteArray()
                    DoubleDataType -> act.data.getDoubleArray().toByteArray()
                    else -> error("Invalid dtype: ${act.dtype}")
                }
            )
            else -> TODO()
        })
        return Transition(
            observation = observation.toTypedObservation() as Observation,
            reward = reward,
            terminated = terminated,
            truncated = truncated,
            info = info.dataMap
        )
    }

    override fun render(): Rendering = env.render().toRendering()

    override fun close() {
        env.close()
    }
}
