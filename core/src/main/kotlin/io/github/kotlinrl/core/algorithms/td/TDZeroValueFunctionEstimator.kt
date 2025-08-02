package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class TDZeroValueFunctionEstimator<State>(
    private val alpha: Double,
    private val gamma: Double
) : TDValueFunctionEstimator<State> {
    override fun estimate(v: ValueFunction<State>, transition: Transition<State, *>): ValueFunction<State> {
        val (s, _, r, sPrime) = transition
        val done = transition.done

        val currentValue = v[s]
        val nextValue = if (done) 0.0 else v[sPrime]
        val updated = currentValue + alpha * (r + gamma * nextValue - currentValue)

        return v.update(s, updated)
    }
}
