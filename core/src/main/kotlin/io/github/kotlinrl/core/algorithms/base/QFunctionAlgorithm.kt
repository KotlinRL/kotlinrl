package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents a reinforcement learning algorithm based on Q-functions. This abstract class extends
 * the base `LearningAlgorithm` and provides additional functionality for handling updates to
 * Q-functions, which are used to evaluate the expected value of taking certain actions in certain states.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial Q-function policy used by the algorithm.
 * @param onPolicyUpdate a callback function invoked when the policy is updated.
 * @param onQFunctionUpdate a callback function invoked when the Q-function is updated.
 */
abstract class QFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    private val onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
) : LearningAlgorithm<State, Action>(initialPolicy, onPolicyUpdate) {
    /**
     * Represents the Q-function currently used by the algorithm, which defines the expected value
     * of taking specific actions in specific states within the environment.
     *
     * The Q-function is an integral part of reinforcement learning algorithms and is updated
     * throughout the learning process to reflect the improved understanding of the environment.
     * Updates to the Q-function are performed via the [onQFunctionUpdate] callback whenever the value changes.
     *
     * This property is initialized with the Q-function from the initial policy and can only be modified
     * within the class or its subclasses.
     */
    var Q = initialPolicy.Q
        protected set(value) {
            field = value
            onQFunctionUpdate(value)
        }
}