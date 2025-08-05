package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class TDValueFunctionPrediction<State, Action>(
    V: ValueFunction<State>,
    private val estimator: TDValueFunctionEstimator<State, Action>,
    private val onValueFunctionUpdate: (ValueFunction<State>) -> Unit = { },
) : TransitionObserver<State, Action> {
    var V = V
        private set

    override fun invoke(transition: Transition<State, Action>) {
        V = estimator.estimate(V, transition)
        onValueFunctionUpdate(V)
    }
}
