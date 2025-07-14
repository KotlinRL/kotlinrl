package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.policy.*

class PolicyAgent<State, Action>(
    override val id: String,
    val policy: Policy<State, Action>,
    val onTrajectory: TrajectoryObserver<State, Action> = TrajectoryObserver {  }
) : Agent<State, Action> {

    override fun act(state: State): Action = policy(state)

    override fun observe(trajectory: Trajectory<State, Action>) = onTrajectory(trajectory)
}
