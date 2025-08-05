package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OffPolicyMonteCarloControl<State, Action>(
    behavioralPolicy: QFunctionPolicy<State, Action>,
    targetPolicy: QFunctionPolicy<State, Action>,
    gamma: Double,
    estimator: TrajectoryQFunctionEstimator<State, Action> = OffPolicyMonteCarloQFunctionEstimator(
        initTargetPolicy = targetPolicy,
        behaviorPolicy = behavioralPolicy,
        gamma = gamma,
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TrajectoryQFunctionAlgorithm<State, Action>(
    initialPolicy = behavioralPolicy,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is OffPolicyMonteCarloQFunctionEstimator -> estimator.targetPolicy =
                it.improve((it as QFunctionPolicy<State, Action>).q)
        }
        onPolicyUpdate(it)
    }
)