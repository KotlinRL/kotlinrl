package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

class NStepExpectedSARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimator: TrajectoryQFunctionEstimator<State, Action> = NStepTDQFunctionEstimator(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepExpectedSARSA()
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {}
) : NStepTD<State, Action>(
    initialPolicy = initialPolicy,
    n = n,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is NStepTDQFunctionEstimator -> estimator.policy = it as QFunctionPolicy<State, Action>
        }
        onPolicyUpdate(it)
    }
)
