package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.Transition
import io.github.kotlinrl.core.TransitionObserver

class TransitionQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: TransitionQFunctionEstimator<State, Action>,
    private val onQFunctionUpdate: (EnumerableQFunction<State, Action>) -> Unit = { }
) : TransitionObserver<State, Action> {

    var q = initialQ
        private set

    override fun invoke(transition: Transition<State, Action>) {
        q = estimator.estimate(q, transition)
        onQFunctionUpdate(q)
    }
}