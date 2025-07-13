package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class TransformReward<
        Observation,
        Action,
        OS : Space<Observation>,
        AS : Space<Action>
        >(
    env: Env<Observation, Action, OS, AS>,
    private val transform: (Double) -> Double
) : SimpleWrapper<Observation, Action, OS, AS>(env) {

    override fun step(action: Action): Transition<Observation> {
        val transition = env.step(action)
        return transition.copy(reward = transform(transition.reward))
    }
}
