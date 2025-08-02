package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.QFunctionAlgorithm
import io.github.kotlinrl.core.algorithms.StateActionKeyFunction

abstract class MonteCarloAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    protected val gamma: Double,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : QFunctionAlgorithm<State, Action>(initialPolicy, initialQ,  onPolicyUpdate, onQFunctionUpdate,improvement)