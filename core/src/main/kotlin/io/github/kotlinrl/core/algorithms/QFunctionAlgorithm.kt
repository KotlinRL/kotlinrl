package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

abstract class QFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    private val onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
) : LearningAlgorithm<State, Action>(initialPolicy, onPolicyUpdate) {
    var q = initialPolicy.q
        protected set(value) {
            field = value
            onQFunctionUpdate(value)
        }
}