package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.td.TDQErrors

/**
 * Implements the SARSA(λ) algorithm, a specific variant of the TD(λ) algorithm that combines
 * the SARSA temporal-difference error calculation mechanism with eligibility traces to update
 * state-action values (Q-values) over multiple time steps.
 *
 * SARSA(λ) is an on-policy reinforcement learning algorithm that updates the Q-function based
 * on the observed transitions and the current policy's action selection. The algorithm computes
 * TD errors using the SARSA update rule, balancing immediate and future rewards while considering
 * the decay of eligibility traces.
 *
 * This class extends `TDLambda` and relies on the SARSA-based TD error function for updates.
 * By default, it employs `ReplacingTrace` as the eligibility trace mechanism, which resets
 * trace values for visited state-action pairs and decays them over time.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param initialPolicy The initial policy used to select actions. This policy is iteratively updated
 *        as the Q-values improve.
 * @param alpha A schedule for the learning rate, controlling the step size in Q-value updates.
 * @param gamma The discount factor used to balance immediate versus future rewards.
 * @param lambda A schedule for the λ parameter, which determines the decay rate of eligibility
 *        traces, influencing the extent to which prior state-action pairs are credited for rewards.
 * @param initialEligibilityTrace The eligibility trace mechanism initialized to track state-action
 *        visitation. Defaults to `ReplacingTrace`, which resets traces to 1 upon visitation and decays them.
 * @param onQFunctionUpdate A callback function invoked after each Q-function update, allowing
 *        custom operations such as logging or monitoring.
 * @param onPolicyUpdate A callback function invoked after a policy update, enabling additional
 *        operations related to updated policies.
 * @param onEligibilityTraceUpdate A callback function invoked during eligibility trace updates,
 *        useful for debugging or analysis of trace dynamics.
 */
class SARSALambda<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
) : TDLambda<State, Action>(initialPolicy, alpha, gamma, lambda, initialEligibilityTrace, onQFunctionUpdate, onPolicyUpdate, onEligibilityTraceUpdate,
    td = TDQErrors.sarsa()
)
