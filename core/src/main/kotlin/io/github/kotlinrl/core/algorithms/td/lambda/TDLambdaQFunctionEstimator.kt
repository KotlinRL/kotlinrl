package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*

class TDLambdaQFunctionEstimator<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val alpha: ParameterSchedule,
    private val lambda: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action>,
    initialEligibilityTrace: EligibilityTrace<State, Action>,
    private val onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { }
) : TransitionQFunctionEstimator<State, Action> {

    var policy: Policy<State, Action> = initialPolicy
    private var trace: EligibilityTrace<State, Action> = initialEligibilityTrace
        set(value) {
            field = value
            onEligibilityTraceUpdate(value)
        }

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a, _, sPrime) = transition
        val done = transition.done
        val aPrime = if (!done) policy(sPrime) else null

        val delta = td(Q, transition, aPrime ?: a, gamma, done)
        trace = trace.decay(gamma, lambda()).update(s, a)
        var updatedQ = Q
        @Suppress("UNCHECKED_CAST")
        trace.values().forEach { (key, traceValue) ->
            val state = when(key.state) {
                is ComparableIntList -> mk.ndarray(key.state.data).asDNArray()
                else -> key.state
            } as State
            val action = key.action as Action
            val newQ = updatedQ[state, action] + alpha() * delta * traceValue
            updatedQ = updatedQ.update(state, action, newQ)
        }

        if (done) trace = trace.clear()

        return updatedQ
    }
}