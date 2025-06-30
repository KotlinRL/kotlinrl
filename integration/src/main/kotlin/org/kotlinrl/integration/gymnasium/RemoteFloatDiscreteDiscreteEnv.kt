package org.kotlinrl.integration.gymnasium

import org.kotlinrl.core.env.*
import org.kotlinrl.open.env.*
import kotlin.random.*

internal class RemoteFloatDiscreteDiscreteEnv(
    envName: String,
    seed: Int? = null,
    render: Boolean = true,
    options: Map<String, String> = emptyMap(),
    host: String = "localhost:50051"
) : FloatDiscreteDiscreteEnv {
    private val env = RemoteEnv(envName, render, options, host)
    override val actionSpace = env.actionSpace.toDiscrete(seed)
    override val observationSpace = env.observationSpace.toDiscrete(seed)
    override val metadata: Map<String, Any> = env.metadata.toMap()
    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun step(act: Int): Transition<Int, Float> {
        val (observation, reward, terminated, truncated, info) = env.step(action(act))
        return Transition(
            observation = observation.int32,
            reward = reward,
            terminated = terminated,
            truncated = truncated,
            info = info.dataMap
        )
    }

    override fun reset(): InitialState<Int> {
        val (observation, info) =  env.reset()
        return InitialState(
            observation = observation.int32,
            info = info.dataMap
        )
    }

    override fun render(): Rendering = env.render().toRendering()

    override fun close() {
        env.close()
    }
}