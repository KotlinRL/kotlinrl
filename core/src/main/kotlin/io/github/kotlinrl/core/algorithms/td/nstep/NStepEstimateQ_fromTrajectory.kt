package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * An implementation of the `EstimateQ_fromTrajectory` interface that utilizes
 * n-step temporal difference (TD) learning to estimate and update a Q-function
 * based on a trajectory of experiences. This class integrates the n-step TD error
 * calculation with adjustable learning parameters.
 *
 * @param State The type representing the environment's state space.
 * @param Action The type representing the actions available in the environment.
 * @param initialPolicy An optional policy governing agent behavior that can guide
 *                      the TD error calculation. Defaults to null if no policy is provided.
 * @param alpha A parameter schedule defining the learning rate to be applied during updates.
 *              The learning rate determines the step size of updates to the Q-function.
 * @param gamma The discount factor applied to future rewards, influencing
 *              the relative weight of immediate versus future rewards in the TD error.
 *              The value must lie within the range [0, 1].
 * @param td A function for computing the n-step TD error used for Q-function updates.
 */
class NStepEstimateQ_fromTrajectory<State, Action>(
    initialPolicy: Policy<State, Action>? = null,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: NStepTDQError<State, Action>
) : EstimateQ_fromTrajectory<State, Action> {

    /**
     * Represents the current policy used in the n-step temporal difference learning
     * algorithm. The policy governs the agent's actions during the learning process
     * and is updated iteratively to improve performance.
     *
     * This variable holds the reference to the policy object and may be modified
     * as a result of updates applied during the learning iterations.
     */
    private var policy = initialPolicy

    /**
     * Represents the action taken at the tail of the current trajectory in n-step temporal difference learning.
     *
     * This variable is used to manage updates in the n-step TD learning process,
     * especially when processing transitions and completing episodes. It helps
     * ensure that the correct action is associated with the tail of the trajectory,
     * facilitating the computation of Q-function updates.
     *
     * It is closely tied to the `estimateQ` logic for trajectory-based Q-function estimation
     * and is dynamically updated based on transitions observed during the learning process.
     */
    var tailAction: Action? = null

    /**
     * Applies the n-step temporal difference (TD) update rule to modify the given Q-function
     * based on the provided trajectory.
     *
     * This method updates the Q-function by computing the n-step TD error and adjusting
     * the Q-value of the first state-action pair in the trajectory. It makes use of a learning rate
     * (alpha), discount factor (gamma), and a policy to perform the updates.
     *
     * @param Q The Q-function representing the estimated action-value for state-action pairs.
     * @param trajectory The trajectory consisting of a sequence of states, actions, and rewards
     *                   used to compute the TD updates. It may represent the agent's experience
     *                   in an environment.
     * @return The updated Q-function after applying the n-step TD update for the given trajectory.
     */
    override operator fun invoke(
        Q: QFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): QFunction<State, Action> {
        if (trajectory.isEmpty()) return Q

        val s0 = trajectory.first().state
        val a0 = trajectory.first().action

        val delta = td(Q, trajectory, policy, tailAction, gamma)
        if (delta == 0.0) return Q
        val updateQ = Q.update(s0, a0, Q[s0, a0] + alpha() * delta)
        policy = policy?.improve(updateQ)
        return updateQ
    }
}
