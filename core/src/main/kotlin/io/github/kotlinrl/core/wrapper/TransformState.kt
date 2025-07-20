package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
class TransformState<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>, WrappedState, WrappedObservationSpace : Space<WrappedState>>(
    env: Env<WrappedState, Action, WrappedObservationSpace, ActionSpace>,
    private val transform: (WrappedState) -> State,
    override val observationSpace: ObservationSpace
) : Wrapper<State, Action, ObservationSpace, ActionSpace, WrappedState, Action, WrappedObservationSpace, ActionSpace>(env) {

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> {
        val initial = env.reset(seed, options)
        return InitialState(state = transform(initial.state), info = initial.info)
    }

    override fun step(action: Action): StepResult<State> {
        val transition = env.step(action)
        return StepResult(
            state = transform(transition.state),
            reward = transition.reward,
            terminated = transition.terminated,
            truncated = transition.truncated,
            info = transition.info
        )
    }

    override val actionSpace: ActionSpace
        get() = env.actionSpace
}

