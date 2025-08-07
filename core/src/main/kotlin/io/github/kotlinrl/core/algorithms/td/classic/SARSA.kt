package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * An implementation of the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm.
 *
 * SARSA is an on-policy learning algorithm, meaning that it updates its Q-function based on the current policy being followed.
 * The algorithm learns by interacting with the environment, observing one-step transitions, and using the temporal difference
 * method to update the Q-function. The Q-function is then used to improve the policy iteratively.
 *
 * Key features of this class:
 * - Integrates on-policy learning with the SARSA update rule.
 * - Allows for flexible customization of learning rate (`alpha`) and discount factor (`gamma`).
 * - Supports dynamic adjustment of Q-function and policy with callbacks (`onQFunctionUpdate` and `onPolicyUpdate`).
 * - By default, uses the `SARSAEstimateQ_fromTransition` implementation for Q-function estimation with SARSA-specific updates.
 *
 * @param State the type representing the state space of the environment.
 * @param Action the type representing the action space of the environment.
 * @param initialPolicy the starting policy that determines the agent's actions at the beginning and is improved iteratively.
 * @param alpha the learning rate, represented as a [ParameterSchedule], which may vary over time.
 * @param gamma the discount factor for future rewards, ranging between 0 and 1; controls the trade-off between immediate and future rewards.
 * @param estimateQ a Q-function estimator that determines how to update the Q-function based on observed transitions, defaulting to SARSA's method.
 * @param onQFunctionUpdate a callback invoked whenever the Q-function is updated; useful for monitoring or logging.
 * @param onPolicyUpdate a callback invoked whenever the policy is updated; allows for external actions on policy changes.
 */
class SARSA<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    estimateQ: EstimateQ_fromTransition<State, Action> = SARSAEstimateQ_fromTransition(alpha, gamma),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionLearningAlgorithm<State, Action>(initialPolicy, estimateQ, onPolicyUpdate, onQFunctionUpdate)



