package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements the n-step Q-Learning temporal-difference reinforcement learning algorithm.
 *
 * This algorithm uses a Q-learning variant of the n-step TD update,
 * which is an off-policy learning approach. It estimates the action-value
 * function by updating Q-values using the maximum value of future states.
 *
 * The algorithm processes agent transitions to calculate n-step updates for the
 * Q-function, attempting to optimize the policy to maximize expected rewards over time.
 *
 * @param initialPolicy The initial policy to be improved by the algorithm.
 * @param alpha A schedule controlling the learning rate for Q-function updates.
 * @param gamma The discount factor for future rewards. Must be in the range [0, 1].
 * @param n The number of steps used for the n-step temporal-difference update.
 * @param estimateQ A trajectory-based Q-function estimator implementing the n-step Q-learning update rule.
 * @param onQFunctionUpdate A callback triggered after updates to the Q-function.
 * @param onPolicyUpdate A callback triggered after updates to the policy.
 */
class NStepQLearning<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimateQ: EstimateQ_fromTrajectory<State, Action> = NStepEstimateQ_fromTrajectory(
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepQLearning()
    ),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {}
) : NStepTD<State, Action>(
    initialPolicy = initialPolicy,
    n = n,
    estimateQ = estimateQ,
    onPolicyUpdate = onPolicyUpdate,
    onQFunctionUpdate = onQFunctionUpdate
)
