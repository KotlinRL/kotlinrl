package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

class QLearning<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    estimator: TransitionQFunctionEstimator<State, Action> = QLearningQFunctionEstimator(alpha, gamma),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate)
