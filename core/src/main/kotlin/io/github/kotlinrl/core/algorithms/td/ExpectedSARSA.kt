package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class ExpectedSARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    estimator: TransitionQFunctionEstimator<State, Action> = ExpectedSARSAQFunctionEstimator(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionQFunctionAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is ExpectedSARSAQFunctionEstimator -> estimator.policy = it as QFunctionPolicy<State, Action>
        }
        onPolicyUpdate(it)
    }
)