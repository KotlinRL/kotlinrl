package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * An implementation of the Off-Policy Monte Carlo Control algorithm for reinforcement learning.
 *
 * This class integrates behavior and target policies to facilitate the learning of an optimal policy
 * by adjusting the Q-function estimates based on trajectories sampled from the environment. The
 * algorithm employs off-policy techniques, such as importance sampling, to reconcile the discrepancies
 * between the behavioral policy and the target policy.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing actions that can be executed in the environment.
 * @param behavioralPolicy the policy under which trajectories are generated. This is used for sampling
 * trajectories during exploration.
 * @param targetPolicy the policy being optimized. The algorithm updates this policy towards optimal performance
 * based on Q-function improvements.
 * @param gamma the discount factor, a value in the range [0, 1], indicating the weight assigned to
 * future rewards during Q-function updates.
 * @param estimateQ the strategy for estimating the Q-function using collected trajectories. The default
 * value is `OffPolicyMonteCarloEstimateQ_fromTrajectory`, which implements an off-policy Monte Carlo approach
 * with importance sampling for correcting policy mismatches.
 * @param onQFunctionUpdate a callback function that is executed on each Q-function update. This allows
 * for additional custom operations to occur during the learning process.
 * @param onPolicyUpdate a callback function that gets invoked upon policy updates. It provides a mechanism
 * to respond to changes in the target policy.
 */
class OffPolicyMonteCarloControl<State, Action>(
    behavioralPolicy: Policy<State, Action>,
    targetPolicy: Policy<State, Action>,
    gamma: Double,
    estimateQ: EstimateQ_fromTrajectory<State, Action> = OffPolicyMonteCarloEstimateQ_fromTrajectory(
        initTargetPolicy = targetPolicy,
        behaviorPolicy = behavioralPolicy,
        gamma = gamma,
    ),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TrajectoryLearningAlgorithm<State, Action>(
    initialPolicy = behavioralPolicy,
    estimateQ = estimateQ,
    onQFunctionUpdate = {
        when (estimateQ) {
            is OffPolicyMonteCarloEstimateQ_fromTrajectory ->
                estimateQ.targetPolicy = estimateQ.targetPolicy.improve(it)
        }
        onQFunctionUpdate(it)
    },
    onPolicyUpdate = onPolicyUpdate
)