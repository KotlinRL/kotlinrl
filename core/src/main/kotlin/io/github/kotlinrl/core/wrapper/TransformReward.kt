package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class TransformReward<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    private val transform: (Double) -> Double
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    override fun step(action: Action): Transition<State> {
        val transition = env.step(action)
        return transition.copy(reward = transform(transition.reward))
    }
}
