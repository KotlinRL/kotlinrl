package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.Transition
import io.github.kotlinrl.core.TransitionObserver

/**
 * A class responsible for predicting and incrementally updating a Q-function based on observed state-action
 * transitions in reinforcement learning. This class leverages a `TransitionQFunctionEstimator` to estimate
 * updated Q-values by incorporating information provided by transitions.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions available in the environment.
 * @param initialQ the initial Q-function, which provides the baseline expected values for state-action pairs.
 * @param estimator the Q-function estimator used to update the Q-function based on observed transitions.
 */
class TransitionQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: TransitionQFunctionEstimator<State, Action>,
) : TransitionObserver<State, Action> {

    /**
     * Represents the Q-function used to estimate the expected values of state-action pairs.
     *
     * This variable is initialized with the `initialQ` provided to the class and serves as the
     * current state of the Q-function throughout the learning process. The Q-function is updated
     * by the `TransitionQFunctionEstimator` when processing transitions observed during learning.
     *
     * Updates to this variable encapsulate changes in the understanding of the expected return
     * for state-action pairs based on observed transitions. It is a core component of reinforcement
     * learning algorithms that leverage Q-functions for decision-making.
     *
     * The setter for this property is private, ensuring it is only modified internally within the class.
     */
    var Q = initialQ
        private set

    /**
     * Invokes a state-action transition, updating the Q-function by incorporating information
     * from the provided transition via the `TransitionQFunctionEstimator`.
     *
     * @param transition the observed transition containing the current state, the action performed,
     * the reward received, and the resulting next state. It is used to estimate and update the Q-function.
     */
    override operator fun invoke(transition: Transition<State, Action>) {
        Q = estimator.estimate(Q, transition)
    }
}