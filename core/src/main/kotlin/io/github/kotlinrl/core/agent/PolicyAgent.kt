package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*

class PolicyAgent<State, Action>(
    override val id: String,
    val policy: Policy<State, Action>,
    val onTransition: ObserveTransition<State, Action>,
    val onTrajectory: ObserveTrajectory<State, Action>
) : Agent<State, Action> {

    override fun act(state: State): Action = policy(state)

    override fun observe(transition: Transition<State, Action>) = onTransition(transition)

    override fun observe(trajectory: List<Transition<State, Action>>, episode: Int) = onTrajectory(trajectory, episode)
}
