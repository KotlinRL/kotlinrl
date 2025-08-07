package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * An implementation of the n-step SARSA reinforcement learning algorithm for policy evaluation
 * and improvement. This class utilizes temporal difference (TD) learning to estimate the
 * action-value function (Q-function) and refine the policy governing an agent's behavior
 * within an environment.
 *
 * n-step SARSA updates involve accumulating rewards over a trajectory of n steps and
 * incorporating the Q-value of the `actual action` taken at the end of the n-step sequence
 * to bootstrap future value estimations. The algorithm operates under the assumption of
 * an on-policy setting, where the policy used for action selection is the same as the
 * policy being improved.
 *
 * This class is highly configurable, offering parameters to control learning rate,
 * discount factor, and trajectory length, as well as callback functions for observing
 * Q-function and policy updates during the learning process.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param initialPolicy The initial policy governing action selection. This policy is
 *                      updated iteratively as learning progresses.
 * @param alpha A schedule defining the learning rate (step size) for Q-function updates.
 *              Controls the magnitude of changes to the Q-function based on TD errors.
 * @param gamma The discount factor applied to future rewards, balancing the importance
 *              of immediate and future rewards. Must lie in the range [0, 1].
 * @param n The number of steps used in the computation of the n-step return. Determines
 *          the trajectory length for TD updates.
 * @param estimateQ An implementation of `EstimateQ_fromTrajectory` that computes the
 *                  Q-function updates based on the provided trajectory.
 *                  Defaults to n-step SARSA-based Q estimation.
 * @param onQFunctionUpdate A callback invoked whenever the Q-function is updated.
 *                          Can be used for logging or monitoring the learning process.
 * @param onPolicyUpdate A callback invoked whenever the policy is updated, allowing
 *                       tracking and analysis of policy changes over time.
 */
class NStepSARSA<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    n: Int,
    estimateQ: EstimateQ_fromTrajectory<State, Action> = NStepEstimateQ_fromTrajectory(
        alpha = alpha,
        gamma = gamma,
        td = NStepTDQErrors.nStepSARSA()
    ),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : NStepTD<State, Action>(initialPolicy, n, estimateQ, onQFunctionUpdate, onPolicyUpdate)