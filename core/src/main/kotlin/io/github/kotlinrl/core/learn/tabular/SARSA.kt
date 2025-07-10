package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.MutableQFunction

class SARSA<State, Action>(
    Q: MutableQFunction<State, Action>,
    alpha: Double,
    gamma: Double
) : TabularTDLearning<State, Action>(Q, alpha, gamma) {

    override fun invoke(experience: Experience<State, Action>) {
        val a = experience.priorAction ?: return
        val aPrime = action ?: error("Next action not yet recorded")
        val s = experience.priorState
        val sPrime = experience.transition.observation
        val r = experience.transition.reward

        val currentValue = Q(s, a)
        val nextValue = if (experience.transition.terminated) 0.0 else Q(sPrime, aPrime)

        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        Q.update(s, a, updated)
    }
}


