package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

abstract class QFunctionAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    protected val onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
) : LearningAlgorithm<State, Action>(initialPolicy, onPolicyUpdate) {
    var q = initialQ
        protected set
}