package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.agent.Transition

abstract class LearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    protected val onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
) {
    operator fun invoke(state: State): Action = policy(state)

    var policy: Policy<State, Action> = initialPolicy
        private set

    fun update(transition: Transition<State, Action>) = observe(transition)

    fun update(trajectory: Trajectory<State, Action>, episode: Int) = observe(trajectory, episode)

    protected open fun observe(transition: Transition<State, Action>) {}

    protected open fun observe(trajectory: Trajectory<State, Action>, episode: Int) {}

    protected fun policyImproved(improvedPolicy: Policy<State, Action>) {
        policy = improvedPolicy
        onPolicyUpdate(policy)
    }
}