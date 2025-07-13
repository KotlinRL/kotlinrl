package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
class TransformObservation<
        Observation,
        Action,
        ObservationSpace : Space<Observation>,
        ActionSpace : Space<Action>,
        WrappedObservation,
        WrappedObservationSpace : Space<WrappedObservation>
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

    override fun step(action: Action): Transition<Observation> {
        val transition = env.step(action)
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

