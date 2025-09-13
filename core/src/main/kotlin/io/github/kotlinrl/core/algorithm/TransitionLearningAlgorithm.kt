package io.github.kotlinrl.core.algorithm

import io.github.kotlinrl.core.PolicyUpdate
import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.api.*
import kotlin.random.*

/**
 * Abstract class defining a transition-based reinforcement learning algorithm.
 *
 * This algorithm relies on processing state-action-reward-next-state transitions
 * to update its policy and Q-function. It extends the `BaseAlgorithm` class, inheriting
 * functionality for managing the learning policy and invoking policy updates.
 *
 * @param State the type representing the state space of the environment.
 * @param Action the type representing the action space of the environment.
 * @param initialPolicy the initial policy used to decide actions based on states.
 * @param onPolicyUpdate a callback invoked upon policy updates.
 * @param rng the random number generator to be used in the algorithm.
 */
abstract class TransitionLearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    rng: Random = Random.Default,
) : BaseAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, rng) {

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