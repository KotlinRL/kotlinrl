package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the Q-Learning algorithm for reinforcement learning. Q-Learning is an off-policy
 * Temporal Difference (TD) learning algorithm that updates the Q-function using the maximum
 * possible reward from the next state-action pair rather than considering the action suggested by
 * the current policy.
 *
 * Q-Learning is suitable for finding optimal policies in environments where all actions may not
 * strictly follow the current policy, as it decouples learning from the policy being followed.
 * This feature makes it an off-policy algorithm and helps in exploring environments effectively.
 *
 * The class relies on a [QLearningQFunctionEstimator] for estimating the TD error, updating the
 * Q-function iteratively with respect to the given discount factor and learning rate. It also
 * facilitates custom invocation of callbacks for when a Q-function or policy update occurs,
 * allowing for more control over the learning and updating processes.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial policy for selecting actions, represented as a Q-function-based policy.
 * @param alpha the learning rate, provided as a [ParameterSchedule] that can adapt over time.
 * @param gamma the discount factor for future rewards, ranging between 0 and 1.
 * @param estimator the transition Q-function estimator responsible for calculating Q-function updates,
 * by default a [QLearningQFunctionEstimator].
 * @param onQFunctionUpdate a callback function invoked whenever the Q-function is updated.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated.
 */
class QLearning<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    estimator: TransitionQFunctionEstimator<State, Action> = QLearningQFunctionEstimator(alpha, gamma),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate)
