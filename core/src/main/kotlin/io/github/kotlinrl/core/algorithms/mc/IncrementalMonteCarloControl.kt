package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.TrajectoryQFunctionAlgorithm
import io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionEstimator

/**
 * Implements incremental Monte Carlo control for reinforcement learning. This class uses trajectory-based
 * Q-function estimation and policy improvement to optimize the agent's decision-making process over time.
 *
 * IncrementalMonteCarloControl is based on the principle of iterative Q-function updates using experience
 * gathered from episodes of interaction with the environment. It employs an incremental update mechanism
 * that allows dynamic adjustments to the Q-function estimates with an optional focus on first-visit updates.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions the agent can take within the environment.
 * @param initialPolicy the initial Q-function policy that dictates the agent's behavior.
 * @param alpha a parameter schedule that determines the step size (learning rate) for Q-function updates.
 *              Defaults to a constant value of 0.05.
 * @param gamma the discount factor used to account for the temporal nature of rewards. Defaults to 0.99.
 * @param firstVisitOnly a flag indicating whether only the first visit to a state-action pair in a trajectory
 *                       should contribute to the Q-function update. Defaults to true.
 * @param estimator the trajectory-based Q-function estimator used to compute Q-function updates. Defaults to
 *                  `IncrementalMonteCarloQFunctionEstimator` configured with `gamma`, `alpha`, and `firstVisitOnly`.
 * @param onQFunctionUpdate a callback function invoked whenever the Q-function is updated. Defaults to an empty lambda.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated. Defaults to an empty lambda.
 */
class IncrementalMonteCarloControl<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true,
    estimator: TrajectoryQFunctionEstimator<State, Action> = IncrementalMonteCarloQFunctionEstimator(
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TrajectoryQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate)
