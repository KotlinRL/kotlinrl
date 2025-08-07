package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements the On-Policy Monte Carlo Control algorithm for reinforcement learning.
 *
 * This class builds on the `TrajectoryLearningAlgorithm` framework and utilizes the
 * On-Policy Monte Carlo method to estimate and improve the Q-function and policy of
 * an agent based on observed trajectories. The algorithm uses sampled episodes to
 * update the Q-function and policy iteratively, with the option to use either the
 * first-visit or every-visit Monte Carlo approach, as indicated by the `firstVisitOnly`
 * parameter.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be taken in the environment.
 * @param initialPolicy the initial policy specifying the agent's action-selection strategy.
 * @param gamma the discount factor that balances the importance of immediate versus future rewards.
 * @param firstVisitOnly a flag indicating whether to consider only the first occurrence of each
 *        state-action pair in the trajectory when updating the Q-function. Defaults to true.
 * @param estimateQ the function for estimating the Q-function from observed trajectories.
 *        By default, this uses the `OnPolicyMonteCarloEstimateQ_fromTrajectory` implementation.
 * @param onQFunctionUpdate a callback triggered when the Q-function is updated. Defaults to a no-op.
 * @param onPolicyUpdate a callback triggered when the policy is updated. Defaults to a no-op.
 */
class OnPolicyMonteCarloControl<State, Action>(
    initialPolicy: Policy<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    estimateQ: EstimateQ_fromTrajectory<State, Action> = OnPolicyMonteCarloEstimateQ_fromTrajectory(
        gamma,
        firstVisitOnly
    ),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { }
) : TrajectoryLearningAlgorithm<State, Action>(initialPolicy, estimateQ, onPolicyUpdate, onQFunctionUpdate)

