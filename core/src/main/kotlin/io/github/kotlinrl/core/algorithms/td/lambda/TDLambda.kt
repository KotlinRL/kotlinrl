package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*

/**
 * An abstract representation of the TD(λ) algorithm, a temporal-difference learning approach
 * that incorporates eligibility traces to efficiently assign credit to prior state-action pairs
 * over multiple time steps. This class serves as a foundation for deriving specific TD(λ) algorithms.
 *
 * TD(λ) enables a balance between Monte Carlo methods and one-step TD methods through the λ parameter.
 * It is highly adaptable to various reinforcement learning scenarios by leveraging customizable
 * components such as eligibility trace implementations, Q-function update rules, and transition estimators.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions that can be taken in the environment.
 * @param initialPolicy The initial policy guiding action selection. This policy can be updated
 *        dynamically to reflect improvements in the Q-function over time.
 * @param alpha A schedule for the learning rate controlling the magnitude of Q-function updates.
 * @param gamma The discount factor used to prioritize immediate versus future rewards during updates.
 * @param lambda A schedule for the λ parameter controlling the decay rate of eligibility traces,
 *        where higher values assign greater weight to distant past state-action pairs.
 * @param initialEligibilityTrace A mechanism for tracking state-action visitations using eligibility traces.
 *        Defaults to `ReplacingTrace`, which resets trace values to 1 upon visitation and decays them over time.
 * @param onQFunctionUpdate A callback invoked after each Q-function update to allow for custom operations
 *        following updates, such as logging or visualization.
 * @param onPolicyUpdate A callback invoked after the policy update to enable responses or additional
 *        operations following policy improvements.
 * @param onEligibilityTraceUpdate A callback invoked after updates to the eligibility trace,
 *        useful for debugging or monitoring the trace dynamics.
 * @param td The temporal-difference (TD) error function defining the mechanism for calculating the
 *        difference between predicted and observed rewards or values.
 * @param estimator The Q-function estimator utilized to predict updates based on observed transitions,
 *        combining TD errors, eligibility traces, and learning parameters.
 */
abstract class TDLambda<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    lambda: ParameterSchedule,
    initialEligibilityTrace: EligibilityTrace<State, Action> = ReplacingTrace(),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
    onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { },
    td: TDQError<State, Action>,
    estimator: TransitionQFunctionEstimator<State, Action> = TDLambdaQFunctionEstimator(
        initialPolicy = initialPolicy,
        alpha = alpha,
        lambda = lambda,
        gamma = gamma,
        td = td,
        initialEligibilityTrace = initialEligibilityTrace,
        onEligibilityTraceUpdate = onEligibilityTraceUpdate
    ),
) : TransitionQFunctionAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is TDLambdaQFunctionEstimator -> estimator.policy = it
        }
        onPolicyUpdate(it)
    }
)