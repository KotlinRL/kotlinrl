package io.github.kotlinrl.core.algorithms.td.et

import io.github.kotlinrl.core.*

class SARSALambda<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
) : TDLambda<State, Action>(initialPolicy, alpha, gamma, lambda, initialEligibilityTrace, onQFunctionUpdate, onPolicyUpdate, onEligibilityTraceUpdate)
