package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class FilterObservation<
        A, AS : Space<A>
        >(
    env: Env<Map<String, Any>, A, Dict, AS>,
    private val keys: Set<String>
) : Wrapper<
        Map<String, Any>, // Output obs is a Map<String, Any>
        A,
        Dict,            // ObservationSpace is a Dict space
        AS,
        Map<String, Any>, // Wrapped obs
        A,
        Dict,
        AS
        >(env) {

    override val observationSpace: Dict by lazy {
        // Only retain specified keys in observationSpace
        val filteredSpaces = env.observationSpace.spaces.filterKeys { it in keys }
        Dict(filteredSpaces)
    }

    override val actionSpace: AS
        get() = env.actionSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<Map<String, Any>> {
        val initial = env.reset(seed, options)
        return InitialState(
            observation = filter(initial.observation),
            info = initial.info
        )
    }

    override fun step(act: A): Transition<Map<String, Any>> {
        val t = env.step(act)
        return t.copy(observation = filter(t.observation))
    }

    private fun filter(obs: Map<String, Any>): Map<String, Any> =
        obs.filterKeys { it in keys }
}
