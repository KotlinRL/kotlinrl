package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the Expected SARSA algorithm for reinforcement learning. This class extends the
 * `TransitionLearningAlgorithm` and provides a framework for learning through state-action
 * transitions while using the Expected SARSA update rule.
 *
 * Expected SARSA is a refinement over traditional SARSA, as it considers the expectation over all
 * possible actions in the next state, weighted by their respective probabilities under the current policy.
 * This reduces variance and improves stability while retaining on-policy characteristics.
 *
 * The algorithm updates its Q-function and improves its policy incrementally based on observed transitions.
 * It integrates components for estimating the Q-values, applying temporal difference updates, and adapting
 * the policy dynamically in response to changes in the Q-function.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be performed within the environment.
 * @param initialPolicy the initial policy governing the agent's decision-making process.
 * @param alpha a [ParameterSchedule] representing the learning rate, which may vary over time.
 * @param gamma the discount factor controlling the weighting of future rewards, constrained between 0 and 1.
 * @param estimateQ the Q-function estimator used to compute updates based on expected rewards and transitions.
 * By default, this uses the Expected SARSA Q-function estimator.
 * @param onQFunctionUpdate a callback invoked after every Q-function update, allowing additional processing
 * or monitoring of changes to the Q-function.
 * @param onPolicyUpdate a callback invoked after every policy update, allowing additional processing or
 * monitoring of changes to the policy.
 */
class ExpectedSARSA<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    estimateQ: EstimateQ_fromTransition<State, Action> = ExpectedSARSAQEstimateQ_fromTransition(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma
    ),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionLearningAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    estimateQ = estimateQ,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimateQ) {
            is ExpectedSARSAQEstimateQ_fromTransition -> estimateQ.policy = it
        }
        onPolicyUpdate(it)
    }
)