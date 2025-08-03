package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

abstract class LearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    protected val onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
) {

    var policy: Policy<State, Action> = initialPolicy
        protected set

    @Suppress("UNCHECKED_CAST")
    protected val improvement = policy as PolicyImprovementStrategy<State, Action>

    operator fun invoke(state: State): Action = policy(state)

    fun update(transition: Transition<State, Action>) = observe(transition)

    fun update(trajectory: Trajectory<State, Action>, episode: Int) = observe(trajectory, episode)

    protected open fun observe(transition: Transition<State, Action>) {}

    protected open fun observe(trajectory: Trajectory<State, Action>, episode: Int) {}
}