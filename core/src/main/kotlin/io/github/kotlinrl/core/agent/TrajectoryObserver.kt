package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*

/**
 * A functional interface for observing trajectories in reinforcement learning environments.
 *
 * The `TrajectoryObserver` is invoked with a trajectory and episode information. It is typically used
 * to process, log, or learn from sequences of transitions that represent an episode in reinforcement learning.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
fun interface TrajectoryObserver<State, Action> {
    /**
     * Handles the observation of a trajectory in a reinforcement learning environment.
     *
     * This function is invoked with a trajectory representing a sequence of transitions observed
     * during an episode and the associated episode identifier. It is used for tasks like logging,
     * analysis, or updating models based on the complete episode.
     *
     * @param trajectory The sequence of transitions observed during an episode.
     *                   Each transition contains information such as the state, action, reward,
     *                   resulting state, and metadata.
     * @param episode The identifier or index of the current episode being observed.
     */
    operator fun invoke(trajectory: Trajectory<State, Action>, episode: Int)
}