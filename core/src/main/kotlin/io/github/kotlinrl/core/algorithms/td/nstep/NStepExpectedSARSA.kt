package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements the n-step Expected SARSA reinforcement learning algorithm.
 *
 * This class utilizes the n-step Expected SARSA approach to estimate and update
 * state-action values (Q-values) through experience, allowing agents to learn
 * optimal policies in a given environment. It combines n-step temporal difference
 * learning and policy-based expected value estimation for efficient updates.
 *
 * @param State The type representing the environment's state space.
 * @param Action The type representing the actions available in the environment.
 * @param initialPolicy The initial policy directing the agent's actions. This policy is
 *                      iteratively improved as the algorithm progresses.
 * @param alpha A parameter schedule defining the learning rate for value function updates.
 *              This rate influences the step size of learning adjustments.
 * @param gamma The discount factor applied to future rewards. Determines the relative
 *              weighting of immediate versus delayed rewards. Must lie in [0, 1].
 * @param n The number of steps to consider in the n-step lookahead used for updates.
 *          Larger values can help capture more meaningful reward trajectories but may
 *          increase computational complexity.
 * @param estimateQ The n-step temporal difference Q-function estimator leveraging a
 *                  trajectory-based approach. It is initialized with a default
 *                  implementation based on n-step Expected SARSA.
 * @param onQFunctionUpdate A callback function triggered after each Q-function update,
 *                          allowing external handling or observation of updated values.
 * @param onPolicyUpdate A callback function executed after each policy update, enabling
 *                       monitoring or modification of the current strategy.
 */
class NStepExpectedSARSA<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimateQ: EstimateQ_fromTrajectory<State, Action> = NStepEstimateQ_fromTrajectory(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepExpectedSARSA()
    ),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {}
) : NStepTD<State, Action>(
    initialPolicy = initialPolicy,
    n = n,
    estimateQ = estimateQ,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = onPolicyUpdate
)
