package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

abstract class QFunctionAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    protected val onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    protected val improvement: PolicyImprovementStrategy<State, Action>
) : LearningAlgorithm<State, Action>(initialPolicy, onPolicyUpdate) {
    var q = initialQ
        private set

    protected fun updatedQFunction(updatedQ: QFunction<State, Action>) {
        q = updatedQ
        onQFunctionUpdate(q)
    }

    protected fun improvePolicy() {
        policyImproved(improvement(q))
    }
}