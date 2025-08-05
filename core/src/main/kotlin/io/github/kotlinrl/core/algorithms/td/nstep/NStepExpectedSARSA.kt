package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements the n-Step Expected SARSA algorithm for reinforcement learning.
 *
 * This algorithm is a temporal difference (TD) method that utilizes an expected update rule
 * across multiple steps to update the Q-function. It combines elements of both on-policy methods
 * (by leveraging the policy's probabilities for updates) and the n-step TD framework for efficient learning.
 *
 * The algorithm uses an `NStepTDQFunctionEstimator` to approximate the Q-function based on the
 * observed trajectory and the policy. The expected value of Q-function updates is computed by
 * averaging over all possible actions weighted by their probabilities under the current policy.
 *
 * Key components:
 * - `initialPolicy`: The initial action-value function policy used for action selection.
 * - `alpha`: The learning rate schedule for Q-function updates.
 * - `gamma`: The discount factor applied to future rewards.
 * - `n`: The number of steps for the n-step TD update.
 * - `estimator`: The Q-function estimator responsible for applying the expected SARSA updates.
 *   Defaults to an `NStepTDQFunctionEstimator` pre-configured for Expected SARSA.
 * - `onQFunctionUpdate`: A callback executed after a Q-function update.
 * - `onPolicyUpdate`: A callback executed after the policy is updated.
 *
 * The algorithm observes trajectories of state-action-reward transitions, performing
 * updates to both the Q-function and policy when these trajectories are processed.
 *
 * This implementation adjusts the current policy within the underlying estimator whenever
 * a policy update occurs, ensuring that action probabilities remain consistent.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions in the environment.
 */
class NStepExpectedSARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimator: TrajectoryQFunctionEstimator<State, Action> = NStepTDQFunctionEstimator(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepExpectedSARSA()
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {}
) : NStepTD<State, Action>(
    initialPolicy = initialPolicy,
    n = n,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is NStepTDQFunctionEstimator -> estimator.policy = it as QFunctionPolicy<State, Action>
        }
        onPolicyUpdate(it)
    }
)
