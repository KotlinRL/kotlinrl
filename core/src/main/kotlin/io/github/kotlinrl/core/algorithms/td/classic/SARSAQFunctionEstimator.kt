package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

class SARSAQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action> = TDQErrors.sarsa()
) : TransitionQFunctionEstimator<State, Action> {
    private var last: Transition<State, Action>? = null

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val prev = last
        last = transition

        if (prev == null) return Q

        val (s, a) = prev
        val (_, aPrime) = transition
        val delta = td(Q, prev, aPrime, gamma, transition.done)
        if (delta == 0.0) return Q
        if (transition.done) last = null

        return Q.update(s, a, Q[s, a] + alpha() * (delta - Q[s, a]))
    }
}