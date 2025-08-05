package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements the n-step SARSA temporal-difference reinforcement learning algorithm.
 *
 * This algorithm is an on-policy learning method that uses the action actually taken
 * by the agent to calculate the n-step TD update. It estimates the action-value function
 * through bootstrap updates that incorporate rewards obtained over n steps
 * and the value of the action taken at step n.
 *
 * The algorithm processes agent transitions to compute n-step updates and improve the policy
 * incrementally to maximize expected rewards in the long term.
 *
 * @param initialPolicy The initial policy that will be iteratively improved as the algorithm runs.
 * @param alpha A schedule defining the learning rate for updates to the Q-function.
 * @param gamma The discount factor applied to future rewards. Must be in the range [0, 1].
 * @param n The number of steps used for the n-step temporal-difference update.
 * @param estimator A trajectory-based Q-function estimator implementing the n-step SARSA update rule.
 * @param onQFunctionUpdate A callback invoked after every Q-function update.
 * @param onPolicyUpdate A callback invoked after every policy update.
 */
class NStepSARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimator: TrajectoryQFunctionEstimator<State, Action> = NStepTDQFunctionEstimator(
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepSARSA()
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : NStepTD<State, Action>(initialPolicy, n, estimator, onQFunctionUpdate, onPolicyUpdate)