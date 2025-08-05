package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

class TDValueFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDVError<State> = TDVErrors.tdZero()
) : TransitionValueFunctionEstimator<State, Action> {

    override fun estimate(
        V: EnumerableValueFunction<State>,
        transition: Transition<State, Action>
    ): EnumerableValueFunction<State> {
        val (s, _, _) = transition
        val delta = td(V, transition, gamma)
        if (delta == 0.0) return V
        return V.update(s, V[s] + alpha() * delta)
    }
}
