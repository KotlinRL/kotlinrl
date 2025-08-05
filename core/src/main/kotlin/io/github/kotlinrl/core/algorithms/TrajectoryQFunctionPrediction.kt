package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.Trajectory
import io.github.kotlinrl.core.TrajectoryObserver

class TrajectoryQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: TrajectoryQFunctionEstimator<State, Action>,
) : TrajectoryObserver<State, Action> {

    var Q = initialQ
        private set

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        Q = estimator.estimate(Q, trajectory)
    }
}