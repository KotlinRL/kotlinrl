package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

abstract class TransitionQFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    estimator: TransitionQFunctionEstimator<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { }
) : QFunctionAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    protected val prediction = TransitionQFunctionPrediction(q, estimator, onQFunctionUpdate)

    override fun observe(transition: Transition<State, Action>) {
        prediction(transition)
        q = prediction.q
        policy = policy.improve(q)
    }
}