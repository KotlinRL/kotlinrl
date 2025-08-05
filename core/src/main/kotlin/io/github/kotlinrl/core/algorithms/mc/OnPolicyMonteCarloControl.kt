package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloControl<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    estimator: TrajectoryQFunctionEstimator<State, Action> = OnPolicyMonteCarloQFunctionEstimator(
        gamma,
        firstVisitOnly
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
) : TrajectoryQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate)

