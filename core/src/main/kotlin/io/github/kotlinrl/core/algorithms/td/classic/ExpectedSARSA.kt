package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the Expected SARSA algorithm for reinforcement learning. This algorithm is a variant
 * of Temporal Difference (TD) learning that updates the Q-function using the expected value of
 * the next state-action pair under the current policy, rather than sampling a specific action.
 *
 * Expected SARSA improves the stability of learning by incorporating the expectation over all
 * possible actions, weighted by their probabilities under the policy, instead of using a single
 * action sampled from the policy (as in SARSA) or the maximum action (as in Q-Learning). This makes
 * it a hybrid between SARSA and Q-Learning, balancing on-policy and off-policy characteristics.
 *
 * The class utilizes an underlying [ExpectedSARSAQFunctionEstimator] to calculate the expected TD error
 * and updates the policy in response to changes in the Q-function. The user can also define custom
 * callbacks for Q-function and policy updates.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial policy for selecting actions, represented as a Q-function-based policy.
 * @param alpha the learning rate, provided as a [ParameterSchedule] that can adapt over time.
 * @param gamma the discount factor for future rewards, ranging between 0 and 1.
 * @param estimator the transition Q-function estimator responsible for calculating the Q-function updates,
 * by default an [ExpectedSARSAQFunctionEstimator].
 * @param onQFunctionUpdate a callback function invoked whenever the Q-function is updated.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated.
 */
class ExpectedSARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    estimator: TransitionQFunctionEstimator<State, Action> = ExpectedSARSAQFunctionEstimator(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionQFunctionAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is ExpectedSARSAQFunctionEstimator -> estimator.policy = it as QFunctionPolicy<State, Action>
        }
        onPolicyUpdate(it)
    }
)