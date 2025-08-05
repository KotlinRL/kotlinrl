package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

abstract class TransitionQFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    estimator: TransitionQFunctionEstimator<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { }
) : QFunctionAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    protected val prediction = TransitionQFunctionPrediction(Q, estimator)

    override fun observe(transition: Transition<State, Action>) {
        prediction(transition)
        Q = prediction.Q
        policy = policy.improve(Q)
    }
}