package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class FilterAction<
        O, OS : Space<O>
        >(
    env: Env<O, Map<String, Any>, OS, Dict>,
    private val keys: Set<String>,
    private val default: Map<String, Any>? = null
) : Wrapper<
        O,
        Map<String, Any>, // Agent provides only selected keys
        OS,
        Dict,
        O,
        Map<String, Any>, // Underlying env expects full Dict action
        OS,
        Dict
        >(env) {

    override val actionSpace: Dict by lazy {
        // Only expose the filtered part to the agent
        val filteredSpaces = env.actionSpace.spaces.filterKeys { it in keys }
        Dict(filteredSpaces)
    }

    override val observationSpace: OS
        get() = env.observationSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> =
        env.reset(seed, options)

    override fun step(act: Map<String, Any>): Transition<O> {
        // Fill missing keys using default (or actionSpace.sample() if not provided)
        val fullAction = buildMap<String, Any> {
            // Agent-controlled keys
            putAll(act)
            // Fill in others
            val allSpaces = env.actionSpace.spaces
            val fillDefault = default ?: env.actionSpace.sample()
            for (k in allSpaces.keys) {
                if (k !in act) {
                    put(k, fillDefault[k] ?: error("No default value for action key $k"))
                }
            }
        }
        return env.step(fullAction)
    }
}
