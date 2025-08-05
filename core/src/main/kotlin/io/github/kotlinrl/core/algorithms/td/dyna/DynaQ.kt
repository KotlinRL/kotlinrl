package io.github.kotlinrl.core.algorithms.td.dyna

import io.github.kotlinrl.core.*

class DynaQ<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    model: LearnableMDPModel<State, Action>,
    planningSteps: Int = 5,
    estimator: TransitionQFunctionEstimator<State, Action> = DynaQEstimator(alpha, gamma, model, planningSteps),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate,onQFunctionUpdate)
