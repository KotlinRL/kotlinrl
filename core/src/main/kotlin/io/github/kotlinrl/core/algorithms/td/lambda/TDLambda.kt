package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*

abstract class TDLambda<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
    td: TDQError<State, Action>,
    estimator: TransitionQFunctionEstimator<State, Action> = TDLambdaQFunctionEstimator(
        initialPolicy = initialPolicy,
        alpha = alpha,
        lambda = lambda,
        gamma = gamma,
        td = td,
        initialEligibilityTrace = initialEligibilityTrace,
        onEligibilityTraceUpdate = onEligibilityTraceUpdate
    ),
) : TransitionQFunctionAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is TDLambdaQFunctionEstimator -> estimator.policy = it
        }
        onPolicyUpdate(it)
    }
)