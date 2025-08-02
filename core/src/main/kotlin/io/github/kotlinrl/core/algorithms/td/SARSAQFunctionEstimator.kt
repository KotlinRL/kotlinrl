package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class SARSAQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TDQFunctionEstimator<State, Action>  {
    private var last: Transition<State, Action>? = null

    override fun estimate(q: QFunction<State, Action>, transition: Transition<State, Action>): QFunction<State, Action> {
        val prev = last
        last = transition

        if (prev == null) return q

        val (s, a) = prev
        val (sPrime, aPrime, r) = transition

        val currentValue = q[s, a]
        val nextValue = if (transition.done) 0.0 else q[sPrime, aPrime]

        val target = r + gamma * nextValue
        val updated = currentValue + alpha() * (target - currentValue)
        if (transition.done) last = null

        return q.update(s, a, updated)
    }
}