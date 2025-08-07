package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*

/**
 * Implements the TD(λ) reinforcement learning algorithm, an extension of Temporal Difference (TD) learning,
 * combining TD methods with eligibility traces. The algorithm allows credit assignment across multiple
 * time steps by maintaining a trace of visited states and actions.
 *
 * TD(λ) balances between one-step and multi-step updates in reinforcement learning through the λ parameter.
 * It can be especially effective in environments with sequential dependencies by propagating rewards over
 * a temporal chain of state-action pairs.
 *
 * This abstract class serves as the foundation for concrete implementations that utilize the TD(λ) learning
 * framework, providing mechanisms for Q-function estimation, policy updates, and eligibility trace management.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @property initialPolicy The initial policy used for selecting actions.
 * @property alpha A parameter schedule for the learning rate, controlling the magnitude of Q-value updates.
 * @property gamma The discount factor, determining the importance of future rewards in the learning process.
 * @property lambda A parameter schedule for the eligibility trace decay factor.
 * @property initialEligibilityTrace The initial eligibility trace, which determines the initial credit assignment dynamics.
 * @property onQFunctionUpdate A callback triggered whenever the Q-function is updated.
 * @property onPolicyUpdate A callback triggered whenever the policy is updated.
 * @property onEligibilityTraceUpdate A callback triggered whenever the eligibility trace is updated.
 * @property td A function to calculate the temporal-difference (TD) error.
 * @property estimateQ The Q-function estimator to calculate and update Q-values based on state transitions,
 * implementing the core of the TD(λ) algorithm.
 */
abstract class TDLambda<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
    td: TDQError<State, Action>,
    estimateQ: EstimateQ_fromTransition<State, Action> = TDLambdaEstimateQ_fromTransition(
        initialPolicy = initialPolicy,
        alpha = alpha,
        lambda = lambda,
        gamma = gamma,
        td = td,
        initialEligibilityTrace = initialEligibilityTrace,
        onEligibilityTraceUpdate = onEligibilityTraceUpdate
    ),
) : TransitionLearningAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    estimateQ = estimateQ,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)