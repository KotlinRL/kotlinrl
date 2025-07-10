package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.space.*
import kotlin.random.*

interface Env<Observation, Action, ObservationSpace : Space<Observation>, ActionSpace : Space<Action>> {
    fun step(act: Action): Transition<Observation>
    fun reset(seed: Int? = null, options: Map<String, String>? = null): InitialState<Observation>
    fun render(): Rendering
    fun close()
    val metadata: Map<String, Any>
    val observationSpace: ObservationSpace
    val actionSpace: ActionSpace
    val random: Random
}
