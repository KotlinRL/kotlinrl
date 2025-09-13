package io.github.kotlinrl.core.algorithm

import io.github.kotlinrl.core.PolicyUpdate
import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.api.*
import kotlin.random.*

/**
 * Abstract base class for trajectory-learning algorithms in reinforcement learning.
 *
 * Trajectory-learning algorithms focus on processing entire sequences of state-action-reward
 * transitions (trajectories) to update their internal models and policies. This class serves
 * as a foundation for implementing such algorithms by providing default behaviors and methods
 * that can be overridden by subclasses to accommodate specific learning mechanisms.
 *
 * @param State the type representing the state space of the environment.
 * @param Action the type representing the action space of the environment.
 * @param initialPolicy the initial policy used to decide actions based on states.
 * @param onPolicyUpdate a callback that is triggered whenever the policy is updated.
 * By default, it performs no operation but can be customized.
 * @param rng a random number generator used for stochastic processes. Defaults to Random.Default.
 */
abstract class TrajectoryLearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    rng: Random = Random.Default,
) : BaseAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, rng) {

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