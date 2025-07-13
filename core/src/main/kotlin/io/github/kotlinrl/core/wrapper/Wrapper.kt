package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import kotlin.random.*

abstract class Wrapper<
        Observation,
        Action,
        ObservationSpace : Space<Observation>,
        ActionSpace : Space<Action>,
        WrappedObservation,
        WrappedAction,
        WrappedObservationSpace : Space<WrappedObservation>,
        WrappedActionSpace : Space<WrappedAction>
        >(
    protected val env: Env<WrappedObservation, WrappedAction, WrappedObservationSpace, WrappedActionSpace>
) : Env<Observation, Action, ObservationSpace, ActionSpace> {

    abstract override fun step(action: Action): Transition<Observation>

    abstract override fun reset(seed: Int?, options: Map<String, String>?): InitialState<Observation>

    override fun render(): Rendering = env.render()

    override fun close() = env.close()

    override val metadata: Map<String, Any>
        get() = env.metadata

    abstract override val observationSpace: ObservationSpace

    abstract override val actionSpace: ActionSpace

    override val random: Random
        get() = env.random
}
