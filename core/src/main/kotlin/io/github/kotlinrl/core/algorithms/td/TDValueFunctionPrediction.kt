package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class TDValueFunctionPrediction<State, Action>(
    v: ValueFunction<State>,
    private val estimator: TDValueFunctionEstimator<State, Action>,
    private val onValueFunctionUpdate: (ValueFunction<State>) -> Unit = { },
) : TransitionObserver<State, Action> {
    var valueFunction = v
        private set

    override fun invoke(transition: Transition<State, Action>) {
        valueFunction = estimator.estimate(valueFunction, transition)
        onValueFunctionUpdate(valueFunction)
    }
}
