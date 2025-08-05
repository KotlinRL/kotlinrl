package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.td.TDQErrors

/**
 * Implements the SARSA(λ) algorithm, a variant of the SARSA algorithm that incorporates
 * eligibility traces to improve temporal credit assignment and learning efficiency.
 *
 * SARSA(λ) builds upon the standard SARSA algorithm by introducing a decay mechanism
 * that assigns credit to multiple state-action pairs based on their contribution to the
 * observed temporal-difference error over time. It is an on-policy, model-free method
 * for reinforcement learning.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions that can be taken in the environment.
 * @param initialPolicy The initial policy used for action selection during interaction with the environment.
 *        This policy is adaptable and will be updated as the Q-function improves.
 * @param alpha A schedule for the learning rate, controlling the step size for Q-value updates.
 * @param gamma The discount factor, determining the weight of future rewards in the Q-value computation.
 * @param lambda A schedule for the λ parameter, which governs the decay of eligibility traces.
 *        Higher values place more emphasis on distant past events.
 * @param initialEligibilityTrace A structure to track the relevance of state-action pairs during training,
 *        with a default implementation that uses replacing traces.
 * @param onQFunctionUpdate A callback invoked after every update to the Q-function. Useful for tracking
 *        updates, debugging, or other custom logic.
 * @param onPolicyUpdate A callback triggered whenever the policy is updated in response to improved Q-values.
 *        This allows integration with external systems or logging mechanisms.
 * @param onEligibilityTraceUpdate A callback invoked upon an update to the eligibility trace structure. This provides
 *        a means to monitor or log the dynamics of the eligibility traces over time.
 */
class SARSALambda<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
) : TDLambda<State, Action>(initialPolicy, alpha, gamma, lambda, initialEligibilityTrace, onQFunctionUpdate, onPolicyUpdate, onEligibilityTraceUpdate,
    td = TDQErrors.sarsa()
)
