package io.github.kotlinrl.core.algorithm

import io.github.kotlinrl.core.agent.*

/**
 * Represents a contract for reinforcement learning algorithms.
 *
 * This interface defines the essential operations required for a learning algorithm,
 * including the ability to determine actions based on states, update the algorithm
 * using individual transitions, and update it using complete trajectories. Implementations
 * of this interface are expected to provide the logic for policy adaptation and learning
 * from interactions with an environment.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken within the environment.
 */
interface LearningAlgorithm<State, Action> {
    /**
     * Determines the action to be taken based on the provided state using the current policy.
     *
     * @param state the current state of the environment.
     * @return the action to be performed for the given state.
     */
    operator fun invoke(state: State): Action

    /**
     * Updates the learning algorithm using the specified transition.
     *
     * This method processes the given state-action-reward-state transition to adapt
     * the algorithm's policy and Q-function, facilitating learning and optimization
     * based on the observed interaction with the environment.
     *
     * @param transition the transition consisting of the current state,
     *        the action taken, the resulting reward, and the next state observed.
     */
    fun update(transition: Transition<State, Action>)

    /**
     * Updates the learning algorithm using the provided trajectory and episode information.
     *
     * This method processes the sequence of state-action-reward transitions for an entire episode,
     * enabling the algorithm to adapt its policy and Q-function based on the observations within
     * the trajectory. The episode information may assist in applying specific strategies or adjustments
     * tied to the progression of learning.
     *
     * @param trajectory the sequence of state-action-reward transitions recorded during an episode.
     * @param episode the number or index representing the specific episode associated with the trajectory.
     */
    fun update(trajectory: Trajectory<State, Action>, episode: Int)
}