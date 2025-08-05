package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * An abstract Q-function-based reinforcement learning algorithm that leverages state-action
 * transitions to update and improve its Q-function and policy. This class extends
 * [QFunctionAlgorithm] and integrates transition-based Q-function predictions and updates.
 *
 * The class is responsible for observing transitions within the environment, predicting updated
 * Q-function values using a [TransitionQFunctionEstimator], and improving the policy based on
 * the updated Q-function. It ensures sequential updates to both the Q-function and the policy
 * to optimize decision-making through reinforcement learning.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial Q-function-based policy used by the algorithm.
 * @param estimator the transition Q-function estimator used to predict the Q-function value updates.
 * @param onPolicyUpdate a callback invoked when the policy is updated.
 * @param onQFunctionUpdate a callback invoked when the Q-function is updated.
 */
abstract class TransitionQFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    estimator: TransitionQFunctionEstimator<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { }
) : QFunctionAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    /**
     * Handles predictions for updating the Q-function based on observed state-action
     * transitions in the environment. This variable utilizes a [TransitionQFunctionPrediction],
     * which employs a given [TransitionQFunctionEstimator] to estimate new Q-function values
     * based on transitions.
     *
     * The predictions are instrumental in sequentially improving the Q-function and policy
     * within the reinforcement learning algorithm. It integrates the current Q-function
     * and the estimator to compute updated values whenever a transition is observed.
     */
    protected val prediction = TransitionQFunctionPrediction(Q, estimator)

    /**
     * Observes a transition within the learning algorithm and updates the internal Q-function and policy.
     *
     * This method processes the observed transition, which encapsulates the state, action,
     * reward, and resulting new state. It updates the Q-function using the `prediction`
     * function and refines the policy based on the improved Q-function.
     *
     * @param transition the observed transition consisting of the initial state,
     * the action taken, the obtained reward, and the resulting state.
     */
    override fun observe(transition: Transition<State, Action>) {
        prediction(transition)
        Q = prediction.Q
        policy = policy.improve(Q)
    }
}