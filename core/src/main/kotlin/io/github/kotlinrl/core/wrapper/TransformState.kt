package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
class TransformState<
        State,
        Action,
        StateSpace : Space<State>,
        ActionSpace : Space<Action>,
        WrappedState,
        WrappedStateSpace : Space<WrappedState>
        >(
    env: Env<WrappedState, Action, WrappedStateSpace, ActionSpace>,
    private val transform: (WrappedState) -> State,
    override val observationSpace: StateSpace
) : Wrapper<
        State,
        Action,
        StateSpace,
        ActionSpace,
        WrappedState,
        Action,
        WrappedStateSpace,
        ActionSpace
        >(env) {

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> {
        val initial = env.reset(seed, options)
        return InitialState(state = transform(initial.state), info = initial.info)
    }

    override fun step(action: Action): Transition<State> {
        val transition = env.step(action)
        return Transition(
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

