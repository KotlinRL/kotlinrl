package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class TransformObservation<
        WrappedObservation,
        Observation,
        Action,
        WrappedObservationSpace : Space<WrappedObservation>,
        ActionSpace : Space<Action>,
        ObservationSpace : Space<Observation>
        >(
    env: Env<WrappedObservation, Action, WrappedObservationSpace, ActionSpace>,
    private val transform: (WrappedObservation) -> Observation,
    override val observationSpace: ObservationSpace
) : Wrapper<
        Observation,
        Action,
        ObservationSpace,
        ActionSpace,
        WrappedObservation,
        Action,
        WrappedObservationSpace,
        ActionSpace
        >(env) {

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<Observation> {
        val initial = env.reset(seed, options)
        return InitialState(observation = transform(initial.observation), info = initial.info)
    }

    override fun step(act: Action): Transition<Observation> {
        val transition = env.step(act)
        return Transition(
            observation = transform(transition.observation),
            reward = transition.reward,
            terminated = transition.terminated,
            truncated = transition.truncated,
            info = transition.info
        )
    }

    override val actionSpace: ActionSpace
        get() = env.actionSpace
}
