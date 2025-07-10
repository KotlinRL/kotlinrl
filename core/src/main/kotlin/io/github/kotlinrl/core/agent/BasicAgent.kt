package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.policy.*

class BasicAgent<State, Action>(
    override val id: String,
    val policy: Policy<State, Action>,
    val onExperience: ExperienceObserver<State, Action>
) : Agent<State, Action> {

    override fun act(state: State): Action = policy(state)

    override fun observe(experience: Experience<State, Action>) = onExperience(experience)
}
