package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

class TransformReward<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    private val transform: (StepResult<State>) -> Double
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    override fun step(action: Action): StepResult<State> {
        val stepResult = env.step(action)
        return stepResult.copy(reward = transform(stepResult))
    }
}
