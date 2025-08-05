package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class ExpectedSARSAQFunctionEstimator<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    initialTDError: TDError<State, Action> = TDErrors.expectedSarsa(initialPolicy)
) : TransitionQFunctionEstimator<State, Action> {
    private var tdError: TDError<State, Action> = initialTDError

    var policy = initialPolicy
        set(value) {
            field = value
            tdError = TDErrors.expectedSarsa(field)
        }

    override fun estimate(
        q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val done = transition.done
        val delta = tdError(q, transition, null, gamma, done)
        val updatedQ = q[s, a] + alpha() * delta
        return q.update(s, a, updatedQ)
    }
}