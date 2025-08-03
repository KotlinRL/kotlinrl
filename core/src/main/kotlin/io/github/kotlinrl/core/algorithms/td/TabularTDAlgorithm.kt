package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

abstract class TabularTDAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    protected val alpha: ParameterSchedule,
    protected val gamma: Double,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : QFunctionAlgorithm<State, Action>(initialPolicy, initialQ, onPolicyUpdate, onQFunctionUpdate)