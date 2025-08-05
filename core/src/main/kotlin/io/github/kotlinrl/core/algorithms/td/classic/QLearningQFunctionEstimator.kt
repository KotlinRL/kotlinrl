package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

class QLearningQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action> = TDQErrors.qLearning()
) : TransitionQFunctionEstimator<State, Action> {
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val delta = td(Q, transition, null, gamma, transition.done)
        if (delta == 0.0) return Q
        return Q.update(s, a, Q[s, a] + alpha() * (delta - Q[s, a]))
    }
}
