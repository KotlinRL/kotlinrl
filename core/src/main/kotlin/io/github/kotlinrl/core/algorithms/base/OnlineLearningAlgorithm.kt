package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents an online learning algorithm that supports updating reinforcement learning models through
 * individual state-action-reward transitions or entire trajectories of interactions. The algorithm acts
 * as a hybrid model, leveraging either transition-based or trajectory-based learning depending on the
 * provided sub-algorithms.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing possible actions within the environment.
 * @param initialPolicy the initial policy used for determining actions based on states. It can be updated
 * during the learning process.
 * @param trajectoryLearningAlgorithm an optional sub-algorithm used for updating the model through
 * trajectory-based learning when trajectory data is provided.
 * @param transitionLearningAlgorithm an optional sub-algorithm used for updating the model through
 * transition-based learning when individual transitions are provided.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated during learning.
 * @param onQFunctionUpdate a callback function invoked whenever the Q-function is updated during learning.
 */
open class OnlineLearningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val trajectoryLearningAlgorithm: TrajectoryLearningAlgorithm<State, Action>? = null,
    private val transitionLearningAlgorithm: TransitionLearningAlgorithm<State, Action>? = null,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { }
) : BaseAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    /**
     * Updates the learning algorithm using the provided state-action transition.
     *
     * This method processes the given transition to refine the Q-function and improve
     * the policy. The transition is utilized to guide the learning process by updating
     * the model through the transition-learning algorithm, if it is defined.
     *
     * @param transition the state-action transition containing the current state, the performed
     *                   action, the resulting reward, and the next state. It is used to derive
     *                   updates for the learning process.
     */
    override fun update(transition: Transition<State, Action>) {
        transitionLearningAlgorithm?.update(transition)
    }

    /**
     * Updates the learning algorithm using the provided trajectory and episode details.
     *
     * This method delegates the update process to the underlying trajectory learning
     * algorithm, if it is defined. The trajectory represents a sequence of state-action-reward
     * transitions observed during an interaction with the environment. The episode number
     * provides context for the learning process and may influence how the trajectory is processed.
     *
     * @param trajectory the sequence of state-action-reward transitions recorded during
     *        an episode, representing the learning experience.
     * @param episode the index or identifier of the episode associated with the provided trajectory.
     */
    override fun update(trajectory: Trajectory<State, Action>, episode: Int) {
        trajectoryLearningAlgorithm?.update(trajectory, episode)
    }
}