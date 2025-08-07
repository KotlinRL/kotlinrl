package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.TrajectoryLearningAlgorithm
import io.github.kotlinrl.core.algorithms.base.EstimateQ_fromTrajectory

/**
 * Implements an incremental Monte Carlo control algorithm for reinforcement learning.
 *
 * This class combines policy improvement and value function estimation, refining a policy
 * iteratively by evaluating it based on state-action trajectories and updating the
 * Q-function incrementally. The algorithm processes complete episodes, making updates
 * based on observed returns and adjusting the policy accordingly.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be taken in the environment.
 * @param initialPolicy the starting policy that determines the agent's initial behavior in the environment.
 * @param alpha a parameter schedule defining the learning rate for incremental updates to Q-values.
 *        Defaults to a constant value of 0.05.
 * @param gamma the discount factor controlling the weight of future rewards, with a default value of 0.99.
 * @param firstVisitOnly a Boolean indicating whether to update Q-values only for the first occurrence of state-action
 *        pairs in a trajectory. Defaults to true.
 * @param estimateQ the Q-function estimation algorithm used to update the Q-values based on observed trajectories.
 *        Defaults to IncrementalMonteCarloEstimateQ_fromTrajectory with the given parameters.
 * @param onQFunctionUpdate a callback triggered after each Q-function update, allowing for custom actions or monitoring
 *        during the learning process. Defaults to a no-op.
 * @param onPolicyUpdate a callback triggered after a policy update, useful for custom actions or monitoring during
 *        the learning process. Defaults to a no-op.
 */
class IncrementalMonteCarloControl<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true,
    estimateQ: EstimateQ_fromTrajectory<State, Action> = IncrementalMonteCarloEstimateQ_fromTrajectory(
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly
    ),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TrajectoryLearningAlgorithm<State, Action>(initialPolicy, estimateQ, onPolicyUpdate, onQFunctionUpdate)
