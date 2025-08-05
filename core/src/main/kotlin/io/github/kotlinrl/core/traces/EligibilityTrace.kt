package io.github.kotlinrl.core.traces

import io.github.kotlinrl.core.*


interface EligibilityTrace<State, Action> {
    fun update(state: State, action: Action): EligibilityTrace<State, Action>
    fun decay(gamma: Double, lambda: Double): EligibilityTrace<State, Action>
    fun values(): Map<StateActionKey<*, *>, Double>
    fun clear(): EligibilityTrace<State, Action>
}