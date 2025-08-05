package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.td.TDErrors.sarsa

class SARSAQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val tdError: TDError<State, Action> = sarsa()
) : TransitionQFunctionEstimator<State, Action> {
    private var last: Transition<State, Action>? = null

    override fun estimate(
        q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val prev = last
        last = transition

        if (prev == null) return q

        val (s, a) = prev
        val (_, aPrime) = transition
        val delta = tdError(q, prev, aPrime, gamma, transition.done)
        val updatedQ = q[s, a] + alpha() * (delta - q[s, a])
        if (transition.done) last = null

        return q.update(s, a, updatedQ)
    }
}