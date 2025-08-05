package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

abstract class TrajectoryQFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    estimator: TrajectoryQFunctionEstimator<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { }
) : QFunctionAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    protected val prediction = TrajectoryQFunctionPrediction(Q, estimator)

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        prediction(trajectory, episode)
        Q = prediction.Q
        policy = policy.improve(Q)
    }
}