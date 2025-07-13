package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class FilterAction<
        State, StateSpace : Space<State>
        >(
    env: Env<State, Map<String, Any>, StateSpace, Dict>,
    private val keys: Set<String>,
    private val default: Map<String, Any>? = null
) : Wrapper<
        State,
        Map<String, Any>, // Agent provides only selected keys
        StateSpace,
        Dict,
        State,
        Map<String, Any>, // Underlying env expects full Dict action
        StateSpace,
        Dict
        >(env) {

    override val actionSpace: Dict by lazy {
        // Only expose the filtered part to the agent
        val filteredSpaces = env.actionSpace.spaces.filterKeys { it in keys }
        Dict(filteredSpaces)
    }

    override val observationSpace: StateSpace
        get() = env.observationSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> =
        env.reset(seed, options)

    override fun step(action: Map<String, Any>): Transition<State> {
        // Fill missing keys using default (or actionSpace.sample() if not provided)
        val fullAction = buildMap<String, Any> {
            // Agent-controlled keys
            putAll(action)
            // Fill in others
            val allSpaces = env.actionSpace.spaces
            val fillDefault = default ?: env.actionSpace.sample()
            for (k in allSpaces.keys) {
                if (k !in action) {
                    put(k, fillDefault[k] ?: error("No default value for action key $k"))
                }
            }
        }
        return env.step(fullAction)
    }
}
