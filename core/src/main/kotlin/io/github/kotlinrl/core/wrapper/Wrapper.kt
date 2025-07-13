package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import kotlin.random.*

abstract class Wrapper<
        State,
        Action,
        ObservationSpace : Space<State>,
        ActionSpace : Space<Action>,
        WrappedState,
        WrappedAction,
        WrappedObservationSpace : Space<WrappedState>,
        WrappedActionSpace : Space<WrappedAction>
        >(
    protected val env: Env<WrappedState, WrappedAction, WrappedObservationSpace, WrappedActionSpace>
) : Env<State, Action, ObservationSpace, ActionSpace> {

    abstract override fun step(action: Action): Transition<State>

    abstract override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State>

    override fun render(): Rendering = env.render()

    override fun close() = env.close()

    override val metadata: Map<String, Any>
        get() = env.metadata

    abstract override val observationSpace: ObservationSpace

    abstract override val actionSpace: ActionSpace

    override val random: Random
        get() = env.random
}
