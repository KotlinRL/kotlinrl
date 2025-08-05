package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

class NStepQLearning<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimator: TrajectoryQFunctionEstimator<State, Action> = NStepTDQFunctionEstimator(
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepQLearning()
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {}
) : NStepTD<State, Action>(
    initialPolicy = initialPolicy,
    n = n,
    estimator = estimator,
    onPolicyUpdate = onPolicyUpdate,
    onQFunctionUpdate = onQFunctionUpdate
)
