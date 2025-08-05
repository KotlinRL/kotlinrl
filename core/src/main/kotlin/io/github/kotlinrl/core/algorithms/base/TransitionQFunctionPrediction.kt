package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.Transition
import io.github.kotlinrl.core.TransitionObserver

class TransitionQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: TransitionQFunctionEstimator<State, Action>,
) : TransitionObserver<State, Action> {

    var Q = initialQ
        private set

    override operator fun invoke(transition: Transition<State, Action>) {
        Q = estimator.estimate(Q, transition)
    }
}