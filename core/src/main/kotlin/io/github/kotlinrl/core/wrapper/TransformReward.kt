package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class TransformReward<
        State,
        Action,
        StateSpace : Space<State>,
        ActionSpace : Space<Action>
        >(
    env: Env<State, Action, StateSpace, ActionSpace>,
    private val transform: (Double) -> Double
) : SimpleWrapper<State, Action, StateSpace, ActionSpace>(env) {

    override fun step(action: Action): Transition<State> {
        val transition = env.step(action)
        return transition.copy(reward = transform(transition.reward))
    }
}
