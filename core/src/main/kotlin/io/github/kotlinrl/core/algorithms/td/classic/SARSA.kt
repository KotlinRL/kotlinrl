package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the SARSA (State-Action-Reward-State-Action) algorithm for reinforcement learning.
 * SARSA is an on-policy Temporal Difference (TD) learning algorithm that updates the Q-function
 * using the action selected by the current policy in the subsequent state. The update is based on
 * the observed transition and the next action taken by the agent, making it strictly follow the
 * current policy during learning.
 *
 * The SARSA algorithm is useful in scenarios where the behavior of the learning process needs
 * to align with the policy being followed, as it reduces deviation between learning and acting.
 * This makes it a fully on-policy learning algorithm.
 *
 * The class relies on a [SARSAQFunctionEstimator] for calculating the TD error and updating
 * the Q-function iteratively based on the discount factor and learning rate. It also allows
 * the use of custom callbacks to monitor or react to updates of the Q-function or policy,
 * providing control over the learning process.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial policy for selecting actions, represented as a Q-function-based policy.
 * @param alpha the learning rate, provided as a [ParameterSchedule] that can adapt over time.
 * @param gamma the discount factor for future rewards, ranging between 0 and 1.
 * @param estimator the transition Q-function estimator used to calculate Q-function updates.
 * Defaults to [SARSAQFunctionEstimator] using the learning rate and discount factor.
 * @param onQFunctionUpdate a callback function invoked whenever the Q-function is updated.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated.
 */
class SARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
     estimator: TransitionQFunctionEstimator<State, Action> = SARSAQFunctionEstimator(alpha, gamma),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate)



