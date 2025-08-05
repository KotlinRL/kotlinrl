package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

class ExpectedSARSAQFunctionEstimator<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    initialTD: TDQError<State, Action> = TDQErrors.expectedSarsa(initialPolicy)
) : TransitionQFunctionEstimator<State, Action> {
    private var td: TDQError<State, Action> = initialTD

    var policy = initialPolicy
        set(value) {
            field = value
            td = TDQErrors.expectedSarsa(field)
        }

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val done = transition.done
        val delta = td(Q, transition, null, gamma, done)
        if (delta == 0.0) return Q
        return Q.update(s, a, Q[s, a] + alpha() * delta)
    }
}