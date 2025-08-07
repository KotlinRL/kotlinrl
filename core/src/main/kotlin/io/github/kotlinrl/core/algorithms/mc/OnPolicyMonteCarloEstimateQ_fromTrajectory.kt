package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements the On-Policy Monte Carlo method for estimating the Q-values of state-action
 * pairs using observed trajectories. This technique operates by processing episodes of
 * state-action-reward transitions to iteratively refine the Q-function. Both first-visit
 * and every-visit Monte Carlo approaches are supported, depending on the `firstVisitOnly`
 * parameter.
 *
 * The algorithm computes cumulative returns for state-action pairs by processing the
 * trajectory in reverse order. If `firstVisitOnly` is enabled, it ensures that only the
 * first occurrence of each state-action pair in the trajectory is considered during the update.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions taken by the agent.
 * @param gamma the discount factor that regulates the weighting of future rewards
 *        relative to immediate rewards.
 * @param firstVisitOnly specifies whether to update the Q-function based on the first visit
 *        to a state-action pair only, as opposed to every occurrence within the trajectory.
 */
class OnPolicyMonteCarloEstimateQ_fromTrajectory<State, Action>(
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true,
) : EstimateQ_fromTrajectory<State, Action> {
    private val returns: MutableMap<StateActionKey<*, *>, Int> = mutableMapOf()

    /**
     * Updates a Q-function based on a given trajectory using the On-Policy Monte Carlo method.
     *
     * This method performs a backward iteration over the trajectory, calculating the return (G)
     * for each state-action pair. Depending on the `firstVisitOnly` setting, it may update
     * each state-action pair only on the first visit within the trajectory. The Q-function is
     * updated incrementally using an average of observed returns for each state-action pair.
     *
     * @param Q the current Q-function to be updated.
     * @param trajectory the trajectory of state-action-reward transitions to process.
     * @return an updated Q-function reflecting the calculated returns for the trajectory.
     */
    override operator fun invoke(
        Q: QFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): QFunction<State, Action> {
        val visited = mutableSetOf<StateActionKey<*, *>>()
        var G = 0.0
        var currentQ = Q

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = StateActionKey(s, a)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val count = returns.getOrDefault(key, 0)
            val oldQ = currentQ[s, a]
            val newQ = oldQ + (G - oldQ) / (count + 1)

            returns[key] = count + 1
            currentQ = currentQ.update(s, a, newQ)
        }

        return currentQ
    }
}
