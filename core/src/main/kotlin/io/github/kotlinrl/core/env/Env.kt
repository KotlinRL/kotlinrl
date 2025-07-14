package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.space.*
import kotlin.random.*

interface Env<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>> {
    fun step(action: Action): Transition<State>
    fun reset(seed: Int? = null, options: Map<String, String>? = null): InitialState<State>
    fun render(): Rendering
    fun close()
    val metadata: Map<String, Any>
    val observationSpace: ObservationSpace
    val actionSpace: ActionSpace
    val random: Random
}
