package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.MaxMutableQFunction

class QLearning<State, Action>(
    Q: MaxMutableQFunction<State, Action>,
    alpha: Double,
    gamma: Double
) : TabularTDLearning<State, Action>(Q, alpha, gamma) {

    override fun invoke(experience: Experience<State, Action>) {
        val s = experience.priorState
        val a = experience.priorAction ?: return
        val sPrime = experience.transition.observation
        val r = experience.transition.reward
        val terminated = experience.transition.terminated

        val currentValue = Q(s, a)
        val nextValue = if (terminated) 0.0 else (Q as MaxMutableQFunction<State, Action>).maxValue(sPrime)
        val target = r + gamma * nextValue
        val updated = currentValue + alpha * (target - currentValue)

        Q.update(s, a, updated)
    }
}
