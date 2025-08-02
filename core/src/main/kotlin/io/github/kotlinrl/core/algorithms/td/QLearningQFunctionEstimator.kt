package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class QLearningQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double
) : TDQFunctionEstimator<State, Action> {
    override fun estimate(q: QFunction<State, Action>, transition: Transition<State, Action>): QFunction<State, Action> {
        val (s, a, r, sPrime) = transition
        val done = transition.done

        val currentQ = q[s, a]
        val maxNextQ = if (done) 0.0 else q.maxValue(sPrime)
        val target = r + gamma * maxNextQ

        val updated = currentQ + alpha() * (target - currentQ)
        return q.update(s, a, updated)
    }
}
