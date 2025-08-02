package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

abstract class TabularTDAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    protected val alpha: ParameterSchedule,
    protected val gamma: Double
) : QFunctionAlgorithm<State, Action>(initialPolicy, initialQ, onPolicyUpdate, onQFunctionUpdate, improvement)