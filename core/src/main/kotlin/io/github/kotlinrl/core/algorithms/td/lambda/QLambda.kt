package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*

/**
 * Implements the Q(λ) algorithm, a variant of temporal-difference learning
 * with the incorporation of eligibility traces and an off-policy learning approach.
 *
 * Q(λ) extends the Q-learning algorithm by utilizing eligibility traces, allowing it
 * to allocate credit more effectively across multiple time steps. The algorithm adjusts
 * the Q-value updates to reflect the contribution of different past state-action pairs.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions that can be taken in the environment.
 * @param initialPolicy The initial policy used to guide the selection of actions.
 *        This policy can be updated during training as Q-values improve.
 * @param alpha A schedule for the learning rate, which determines the step size
 *        of Q-value updates.
 * @param gamma The discount factor used to balance short-term and long-term rewards.
 * @param lambda A schedule for the λ parameter, which controls the decay of eligibility
 *        traces. Higher values of λ place more weight on distant past events.
 * @param initialEligibilityTrace An implementation of eligibility traces that tracks
 *        how recently and frequently state-action pairs have been visited.
 *        Defaults to `ReplacingTrace`, which resets trace values to 1 upon visitation.
 * @param onQFunctionUpdate A callback invoked after each Q-value update. This can be used
 *        for logging, monitoring, or custom operations after updates.
 * @param onPolicyUpdate A callback invoked after the policy is updated to reflect the
 *        improved Q-values.
 * @param onEligibilityTraceUpdate A callback invoked after the eligibility trace is
 *        updated. This can be useful for debugging or visualization of traces.
 */
class QLambda<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
) : TDLambda<State, Action>(initialPolicy, alpha, gamma, lambda, initialEligibilityTrace, onQFunctionUpdate, onPolicyUpdate, onEligibilityTraceUpdate,
    td = TDQErrors.qLearning()
)
