package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

class NStepSARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimator: TrajectoryQFunctionEstimator<State, Action> = NStepTDQFunctionEstimator(
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepSARSA()
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : NStepTD<State, Action>(initialPolicy, n, estimator, onQFunctionUpdate, onPolicyUpdate)