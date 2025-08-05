package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

class TDValueFunctionEstimator<State, Action>(
    private val alpha: Double,
    private val gamma: Double,
    private val td: TDVError<State, Action> = TDVErrors.tdZero()
) : TransitionValueFunctionEstimator<State, Action> {

    override fun estimate(
        V: EnumerableValueFunction<State>,
        transition: Transition<State, Action>
    ): EnumerableValueFunction<State> {
        val (s, _, _) = transition
        val delta = td(V, transition, gamma)
        val updated = V[s] + alpha * delta
        return V.update(s, updated)
    }
}
