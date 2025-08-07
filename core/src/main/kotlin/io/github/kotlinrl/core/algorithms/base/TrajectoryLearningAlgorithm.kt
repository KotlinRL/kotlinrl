package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents a reinforcement learning algorithm focused on trajectory-based learning,
 * where learning updates are performed using entire episodes or sequences of state-action-reward
 * transitions (trajectories). The algorithm maintains a policy and Q-function, which are adjusted
 * based on the provided trajectories to improve decision-making over time.
 *
 * This class supports on-policy updates of the policy and Q-function and delegates the
 * Q-function estimation to an implementation of `EstimateQ_fromTrajectory`.
 *
 * @param State the type representing the states within the environment.
 * @param Action the type representing the actions an agent can perform within the environment.
 * @param initialPolicy the initial policy defining the agent's action-selection strategy.
 * @param estimateQ a functional interface used to estimate a new Q-function from a given trajectory.
 * @param onPolicyUpdate a callback triggered when the policy is updated. Default is a no-op.
 * @param onQFunctionUpdate a callback triggered when the Q-function is updated. Default is a no-op.
 */
open class TrajectoryLearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val estimateQ: EstimateQ_fromTrajectory<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { }
) : BaseAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    /**
     * Updates the learning algorithm using the specified state-action-reward transition.
     *
     * This method processes the given transition to adjust the internal state of
     * the algorithm. The actual update logic is delegated to the `observe` method,
     * which subclasses can override to implement specific behavior.
     *
     * @param transition the state-action-reward transition representing an interaction
     * between the agent and the environment, including the current state, the action taken,
     * the resulting reward, and the next state observed.
     */
    override fun update(transition: Transition<State, Action>) = observe(transition)

    /**
     * Updates the learning algorithm using the given trajectory and episode information.
     *
     * The update logic is delegated to the `observe` method, which processes the trajectory
     * and episode to adjust the algorithm's internal state. This method allows the learning
     * algorithm to improve its Q-function and policy based on the observed trajectory for
     * the specified episode.
     *
     * @param trajectory the sequence of state-action-reward transitions representing the agent's
     *        interactions with the environment during an episode.
     * @param episode the index or identifier representing the current episode associated with
     *        the provided trajectory.
     */
    override fun update(trajectory: Trajectory<State, Action>, episode: Int) = observe(trajectory, episode)

    /**
     * Processes a provided trajectory and episode index to update the algorithm's Q-function
     * and improve its policy based on the observed data.
     *
     * This method implements the core logic for refining the quality-function and policy
     * by utilizing the estimateQ function for updating the Q-function and the policy's
     * improve method for policy optimization. It is designed to be open for overriding
     * by subclasses to provide specific behavior for trajectory-based learning algorithms.
     *
     * @param trajectory the sequence of state-action-reward transitions representing
     *        the agent's interactions with the environment during an episode.
     * @param episode the index or identifier of the current episode corresponding to
     *        the provided trajectory.
     */
    protected open fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        Q = estimateQ(Q, trajectory)
        policy = policy.improve(Q)
    }

    /**
     * Since this Learning Algorithm is focused on Trajectory learning, this operation
     * does nothing by default.  Subclasses are free to override provided that they handle
     * estimation of Q accordingly
     *
     * @param transition the state-action-reward transition representing an interaction
     * between the agent and the environment, including the current state, action taken,
     * resulting reward, and next state.
     */
    protected open fun observe(transition: Transition<State, Action>) {}
}