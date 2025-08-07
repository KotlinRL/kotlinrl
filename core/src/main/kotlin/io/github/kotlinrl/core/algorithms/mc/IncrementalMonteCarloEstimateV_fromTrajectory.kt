package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements an incremental Monte Carlo method to estimate the value function (V) from a given trajectory.
 *
 * This class uses trajectory data to iteratively update the value function estimates for states
 * using a combination of Monte Carlo returns and a parameter-driven learning rate. It supports
 * both first-visit and every-visit Monte Carlo methods, determined by the `firstVisitOnly` flag.
 *
 * Key features include:
 * - Handles discounting of future rewards using a specified discount factor (`gamma`).
 * - Adjustment of state-value updates based on a customizable parameter schedule (`alpha`).
 * - Optional usage of first-visit Monte Carlo updates, skipping subsequent visits in a trajectory
 *   for the same state.
 *
 * The algorithm iterates backward through the provided trajectory, accumulating rewards (`G`)
 * and updating the value function for each state visited. For first-visit Monte Carlo, states
 * already visited within the same trajectory are skipped. It provides an efficient way to update
 * value estimates incrementally without requiring storage of all trajectories or state-action pairs.
 *
 * @param State the type representing the states of the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param gamma the discount factor for future rewards, specifying the importance of rewards farther
 *        in the future. Typically lies in the range [0, 1].
 * @param alpha a [ParameterSchedule] for dynamically computing the learning rate used in value updates.
 *        Defaults to a constant schedule with a value of 0.05.
 * @param firstVisitOnly a Boolean flag indicating whether to use first-visit Monte Carlo. If true, only
 *        the first occurrence of a state in a trajectory is considered for updating the value function.
 *        Defaults to true.
 */
class IncrementalMonteCarloEstimateV_fromTrajectory<State, Action>(
    private val gamma: Double,
    private val alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    private val firstVisitOnly: Boolean = true,
) : EstimateV_fromTrajectory<State, Action> {

    /**
     * Updates the given value function using the provided trajectory based on an incremental Monte Carlo method.
     *
     * This method iterates through the trajectory in reverse order, computes the return for each state,
     * and updates the value function accordingly. It supports both first-visit and every-visit updates
     * depending on the configuration. For first-visit updates, states that have already been visited
     * in the trajectory are skipped to avoid redundant updates.
     *
     * The update process utilizes a specified discount factor to compute the returns and applies a
     * learning rate parameter to the incremental updates of state values.
     *
     * @param V the initial value function which maps states to their estimated values.
     * @param trajectory the trajectory consisting of states, actions, and rewards observed during an episode.
     *        Each trajectory element contributes to updating the value function.
     * @return a new value function representing the updated estimates after processing the trajectory.
     */
    override fun invoke(V: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State> {
        val visited = mutableSetOf<Comparable<*>>()
        var G = 0.0
        var newV = V

        for ((s, _, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = s.toComparable()

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val oldV = newV[s]
            val updatedV = oldV + alpha() * (G - oldV)

            newV = newV.update(s, updatedV)
        }

        return newV
    }
}
