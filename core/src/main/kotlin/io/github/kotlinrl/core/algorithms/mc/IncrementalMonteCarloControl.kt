package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.TrajectoryQFunctionAlgorithm
import io.github.kotlinrl.core.algorithms.TrajectoryQFunctionEstimator

class IncrementalMonteCarloControl<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true,
    estimator: TrajectoryQFunctionEstimator<State, Action> = IncrementalMonteCarloQFunctionEstimator(
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TrajectoryQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate)
