package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.td.TDErrors.qLearning

class QLearningQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val tdError: TDError<State, Action> = qLearning()
) : TransitionQFunctionEstimator<State, Action> {
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val delta = tdError(Q, transition, null, gamma, transition.done)
        val updatedQ = Q[s, a] + alpha() * (delta - Q[s, a])
        return Q.update(s, a, updatedQ)
    }
}
