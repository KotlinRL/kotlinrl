package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class TDZeroValueFunctionEstimator<State, Action>(
    private val alpha: Double,
    private val gamma: Double
) : TDValueFunctionEstimator<State, Action> {
    override fun estimate(V: ValueFunction<State>, transition: Transition<State, Action>): ValueFunction<State> {
        val (s, _, r, sPrime) = transition
        val done = transition.done

        val currentV = V[s]
        val nextValue = if (done) 0.0 else V[sPrime]
        val updated = currentV + alpha * (r + gamma * nextValue - currentV)

        return V.update(s, updated)
    }
}
