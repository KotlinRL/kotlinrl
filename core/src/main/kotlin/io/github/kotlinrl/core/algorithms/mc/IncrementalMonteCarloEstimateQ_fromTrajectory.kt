package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements an incremental Monte Carlo method for estimating Q-values from trajectories
 * in reinforcement learning. This class sequentially updates a given Q-function using
 * cumulative rewards calculated from a trajectory, supporting both first-visit and
 * every-visit methodologies.
 *
 * @param State the type representing states within the environment.
 * @param Action the type representing actions that can be performed within the environment.
 * @param gamma the discount factor for future rewards, influencing the weighting of future
 *        returns relative to immediate rewards.
 * @param alpha a schedule defining the learning rate, which determines the step size used
 *        when updating Q-values.
 * @param firstVisitOnly a Boolean flag indicating whether only the first visit to each
 *        state-action pair in the trajectory should be used for updating Q-values. If set
 *        to `false`, every visit will be used.
 */
class IncrementalMonteCarloEstimateQ_fromTrajectory<State, Action>(
    private val gamma: Double,
    private val alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    private val firstVisitOnly: Boolean = true,
) : EstimateQ_fromTrajectory<State, Action> {

    /**
     * Updates a Q-function based on a given trajectory using the incremental Monte Carlo method.
     * This method processes the trajectory in reverse order and updates Q-values for state-action
     * pairs based on cumulative rewards, considering the first-visit or every-visit approach.
     *
     * @param Q the initial Q-function to be updated. Represents the action-value function defining
     *        the quality of state-action pairs.
     * @param trajectory the sequence of state-action-reward tuples (trajectory) used for updating
     *        the Q-function. The trajectory is evaluated in reverse to calculate cumulative rewards.
     * @return the updated Q-function reflecting improvements based on the provided trajectory.
     */
    override operator fun invoke(
        Q: QFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): QFunction<State, Action> {
        var currentQ = Q
        val visited = mutableSetOf<StateActionKey<*, *>>()
        var G = 0.0

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = StateActionKey(s, a)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val oldQ = currentQ[s, a]
            val updatedQ = oldQ + alpha() * (G - oldQ)
            currentQ = currentQ.update(s, a, updatedQ)
        }

        return currentQ
    }
}
