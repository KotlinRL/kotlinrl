package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*

class LearningAgent<State, Action>(
    override val id: String,
    val algorithm: LearningAlgorithm<State, Action>,
) : Agent<State, Action> {

    override fun act(state: State): Action = algorithm(state)

    override fun observe(transition: Transition<State, Action>) = algorithm.update(transition)

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) = algorithm.update(trajectory, episode)
}
