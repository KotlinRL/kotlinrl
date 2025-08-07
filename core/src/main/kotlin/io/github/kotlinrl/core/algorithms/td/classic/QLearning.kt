package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the Q-Learning algorithm, a model-free, off-policy reinforcement learning method.
 *
 * Q-Learning updates the action-value function (Q-function) for each state-action pair based on
 * the reward received and the maximum estimated future rewards, without considering the actions
 * suggested by the current policy. This allows the algorithm to learn an optimal policy by maximizing
 * long-term rewards.
 *
 * The Q-function is updated using the Temporal Difference (TD) learning method, with Q-value updates
 * derived from observed state-action transitions. The formula used for updates is:
 *
 * Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) − Q(s, a)]
 *
 * where:
 * - s: current state
 * - a: current action
 * - r: reward received after executing action a in state s
 * - s': next state
 * - a': next action
 * - α: learning rate (controlled by the [alpha] parameter schedule)
 * - γ: discount factor applied to future rewards
 *
 * This implementation allows customization of Q-function estimation via [estimateQ], enabling
 * flexible behavior for different Q-function update rules. Additionally, callbacks are provided
 * for handling events such as Q-function updates ([onQFunctionUpdate]) and policy updates ([onPolicyUpdate]).
 *
 * @param State the type representing the environment's states.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param initialPolicy the starting policy that governs action selection in the environment.
 * @param alpha a parameter schedule that controls the learning rate for Q-function updates.
 * @param gamma the discount factor, a value between 0 and 1, that determines the weight of future rewards.
 * @param estimateQ an implementation of the Q-function estimator, defaulting to Q-Learning-specific updates.
 * @param onQFunctionUpdate a callback triggered whenever the Q-function is updated.
 * @param onPolicyUpdate a callback triggered whenever the policy is updated.
 */
class QLearning<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    estimateQ: EstimateQ_fromTransition<State, Action> = QLearningEstimateQ_fromTransition(alpha, gamma),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionLearningAlgorithm<State, Action>(initialPolicy, estimateQ, onPolicyUpdate, onQFunctionUpdate)
