package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

class TransitionValueFunctionPrediction<State, Action>(
    initialV: EnumerableValueFunction<State>,
    private val estimator: TransitionValueFunctionEstimator<State, Action>,
) : TransitionObserver<State, Action> {

    var V = initialV
        private set

    override operator fun invoke(transition: Transition<State, Action>) {
        V = estimator.estimate(V, transition)
    }
}