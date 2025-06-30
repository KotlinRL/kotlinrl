package org.kotlinrl.core.env

import org.kotlinrl.core.space.*
import kotlin.random.*

interface Env<Observation, Action, Reward, ObservationSpace : Space<Observation>, ActionSpace : Space<Action>> {
    fun step(act: Action): Transition<Observation, Reward>
    fun reset(): InitialState<Observation>
    fun render(): Rendering
    fun close()
    val metadata: Map<String, Any>
    val observationSpace: ObservationSpace
    val actionSpace: ActionSpace
    val random: Random
}
