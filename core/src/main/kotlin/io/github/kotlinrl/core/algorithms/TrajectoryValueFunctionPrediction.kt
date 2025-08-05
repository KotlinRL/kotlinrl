package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

class TrajectoryValueFunctionPrediction<State, Action>(
    initialV: EnumerableValueFunction<State>,
    private val estimator: TrajectoryValueFunctionEstimator<State, Action>,
) : TrajectoryObserver<State, Action> {

    var V = initialV
        private set

    override operator fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        V = estimator.estimate(V, trajectory)
    }
}