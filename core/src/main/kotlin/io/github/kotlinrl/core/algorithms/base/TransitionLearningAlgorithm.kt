package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * A learning algorithm class for reinforcement learning that abstracts transition-based learning methodologies.
 * It builds upon the base functionality provided by the `BaseAlgorithm` class and uses state-action transitions
 * to iteratively refine both the Q-function and policy.
 *
 * This class emphasizes processing individual transitions over entire trajectories, making it suitable
 * for algorithms that operate incrementally on state-action pairs, rather than on sequences of experiences.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be performed within the environment.
 * @param initialPolicy the starting policy that governs decision-making within the algorithm.
 * @param estimateQ a function to update the Q-function based on observed state-action transitions.
 * @param onPolicyUpdate a callback triggered whenever the policy is updated.
 * @param onQFunctionUpdate a callback triggered whenever the Q-function is updated.
 */
open class TransitionLearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val estimateQ: EstimateQ_fromTransition<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { }
) : BaseAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    /**
     * Updates the learning algorithm using the specified transition.
     *
     * This method processes the given state-action-reward-state transition to adapt
     * the algorithm's policy and Q-function. It delegates to the `observe` method
     * for updating the internal representation of the Q-function and policy, enabling
     * further learning based on the observed transition.
     *
     * @param transition the transition consisting of the current state,
     *                   the action taken, the resulting reward, and the next state observed.
     */
    override fun update(transition: Transition<State, Action>) = observe(transition)

    /**
     * Updates the learning algorithm using the provided trajectory and episode information.
     *
     * This method delegates to the `observe` function, enabling subclasses to implement
     * specific logic for processing a trajectory and updating their Q-function or policy
     * based on the observed state-action-reward transitions.
     *
     * @param trajectory the sequence of state-action-reward transitions gathered during an episode.
     * @param episode the identifier or index of the episode associated with the given trajectory.
     */
    override fun update(trajectory: Trajectory<State, Action>, episode: Int) = observe(trajectory, episode)

    /**
     * Processes a state-action-reward-next-state transition to update the Q-function and improve the policy.
     *
     * This method leverages the provided transition to update the Q-function using the `estimateQ` function
     * and subsequently improves the policy based on the updated Q-function. It is designed to be overridden
     * by subclasses that may provide specialized implementations for handling the learning process.
     *
     * @param transition the state-action-reward-next-state transition used to update the Q-function and policy.
     */
    protected open fun observe(transition: Transition<State, Action>) {
        Q = estimateQ(Q, transition)
        policy = policy.improve(Q)
    }

    /**
     * Since this Learning Algorithm is focused on Transition learning, this operation
     * does nothing by default.  Subclasses are free to override provided that they handle
     * estimation of Q accordingly
     *
     * @param trajectory the sequence of state-action-reward transitions gathered during an episode.
     * @param episode the identifier or index of the episode associated with the given trajectory.
     */
    protected open fun observe(trajectory: Trajectory<State, Action>, episode: Int) {}
}