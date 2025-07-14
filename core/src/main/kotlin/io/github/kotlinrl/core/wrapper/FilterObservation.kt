package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class FilterObservation<Action, ActionSpace : Space<Action>>(
    env: Env<Map<String, Any>, Action, Dict, ActionSpace>,
    private val keys: Set<String>
) : Wrapper<
        Map<String, Any>, // Output obs is a Map<String, Any>
        Action,
        Dict,            // ObservationSpace is a Dict space
        ActionSpace,
        Map<String, Any>, // Wrapped obs
        Action,
        Dict,
        ActionSpace
        >(env) {

    override val observationSpace: Dict by lazy {
        // Only retain specified keys in observationSpace
        val filteredSpaces = env.observationSpace.spaces.filterKeys { it in keys }
        Dict(filteredSpaces)
    }

    override val actionSpace: ActionSpace
        get() = env.actionSpace

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<Map<String, Any>> {
        val initial = env.reset(seed, options)
        return InitialState(
            state = filter(initial.state),
            info = initial.info
        )
    }

    override fun step(action: Action): Transition<Map<String, Any>> {
        val t = env.step(action)
        return t.copy(state = filter(t.state))
    }

    private fun filter(obs: Map<String, Any>): Map<String, Any> =
        obs.filterKeys { it in keys }
}
