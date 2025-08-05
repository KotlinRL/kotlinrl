package io.github.kotlinrl.core.traces

import io.github.kotlinrl.core.*

class ReplacingTrace<State, Action> : EligibilityTrace<State, Action> {
    private val traces = mutableMapOf<StateActionKey<*, *>, Double>()

    override fun update(state: State, action: Action): EligibilityTrace<State, Action> {
        val key = stateActionKey(state, action)
        traces[key] = 1.0
        return this
    }

    override fun decay(gamma: Double, lambda: Double): EligibilityTrace<State, Action> {
        traces.replaceAll { _, value -> gamma * lambda * value }
        return this
    }

    override fun values(): Map<StateActionKey<*, *>, Double> = traces.toMap()


    override fun clear(): EligibilityTrace<State, Action> {
        traces.clear()
        return this
    }
}
