package org.kotlinrl.integration.gymnasium

import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.kotlinrl.core.env.*
import org.kotlinrl.open.env.*
import kotlin.random.*

internal class RemoteBoxNDArrayFloatD1BoxNDArrayFloatD1Env(
    envName: String,
    seed: Int? = null,
    render: Boolean = true,
    options: Map<String, String> = emptyMap(),
    host: String = "localhost:50051"
) : BoxNDArrayFloatD1BoxNDArrayFloatD1Env {
    private val env = RemoteEnv(envName, render, options, host)
    override val observationSpace = env.observationSpace.toBoxNDArrayFloatD1(seed)
    override val actionSpace = env.actionSpace.toBoxNDArrayFloatD1(seed)
    override val metadata: Map<String, Any> = env.metadata.toMap()
    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun step(act: NDArray<Float, D1>): Transition<NDArray<Float, D1>> {
        val (observation, reward, terminated, truncated, info) = env.step(action(act.toNDArray()))
        return Transition(
            observation = observation.array.toNDArrayFloat1(),
            reward = reward,
            terminated = terminated,
            truncated = truncated,
            info = info.dataMap
        )
    }

    override fun reset(): InitialState<NDArray<Float, D1>> {
        val (observation, info) =  env.reset()
        return InitialState(
            observation = observation.array.toNDArrayFloat1(),
            info = info.dataMap
        )
    }

    override fun render(): Rendering = env.render().toRendering()

    override fun close() {
        env.close()
    }
}