package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*

/**
 * Implements the Q(λ) algorithm, a specific variant of the TD(λ) algorithm utilizing Q-Learning for
 * temporal-difference error calculations, combined with eligibility traces for assigning credit
 * to state-action pairs over multiple time steps.
 *
 * Q(λ) combines off-policy learning using the maximum Q-value of the following state with a
 * lambda-return mechanism, enabling a tunable balance between delayed and immediate rewards
 * in reinforcement learning tasks.
 *
 * This class extends `TDLambda` and overrides the temporal-difference error calculation mechanism
 * to use the Q-Learning method. By default, it employs `ReplacingTrace` as the eligibility trace
 * mechanism, ensuring that trace values for visited state-action pairs are reset upon visitation
 * and decay over time.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing possible actions in the environment.
 * @param initialPolicy The initial policy used to guide action selection. This can be improved
 *        iteratively based on updates to the Q-function.
 * @param alpha A schedule for the learning rate, which controls how significantly Q-values
 *        adjust in response to observed transitions.
 * @param gamma Discount factor prioritizing immediate versus delayed rewards in Q-value updates.
 * @param lambda A schedule for the λ parameter that determines the decay rate of eligibility
 *        traces, where higher values emphasize past visitations.
 * @param initialEligibilityTrace The eligibility trace mechanism employed to track state-action
 *        visitation history. Defaults to `ReplacingTrace`.
 * @param onQFunctionUpdate A callback function executed after each Q-function update, enabling
 *        additional actions such as logging or monitoring.
 * @param onPolicyUpdate A callback function executed after policy updates, allowing for extra
 *        operations or monitoring related to policy adjustments.
 * @param onEligibilityTraceUpdate A callback function executed during updates to the eligibility
 *        trace, useful for additional analysis or debugging.
 */
class QLambda<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
) : TDLambda<State, Action>(initialPolicy, alpha, gamma, lambda, initialEligibilityTrace, onQFunctionUpdate, onPolicyUpdate, onEligibilityTraceUpdate,
    td = TDQErrors.qLearning()
)
