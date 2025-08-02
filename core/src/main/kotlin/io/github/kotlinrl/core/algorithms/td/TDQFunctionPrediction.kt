package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class TDQFunctionPrediction<State, Action>(
    initialQ: QFunction<State, Action>,
    private val estimator: TDQFunctionEstimator<State, Action>,
    private val onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { }
) : TransitionObserver<State, Action> {

    var q = initialQ
        private set

    override fun invoke(transition: Transition<State, Action>) {
        q = estimator.estimate(q, transition)
        onQFunctionUpdate(q)
    }
}
