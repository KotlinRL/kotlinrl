package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements an On-Policy Monte Carlo method for estimating the Q-function of
 * a reinforcement learning agent based on its observed trajectories.
 *
 * This class updates the Q-function estimation using the returns calculated from
 * the provided trajectory. Monte Carlo estimation involves averaging observed
 * returns for state-action pairs over multiple episodes. This implementation
 * supports both first-visit and every-visit Monte Carlo methods depending on the
 * `firstVisitOnly` parameter.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be taken in the environment.
 * @param gamma the discount factor that determines the significance of future rewards while calculating returns.
 * @param firstVisitOnly a flag indicating whether to consider only the first occurrence of state-action pairs
 *        in the trajectory when computing their returns. Defaults to true.
 */
class OnPolicyMonteCarloQFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true,
) : TrajectoryQFunctionEstimator<State, Action> {
    private val returns: MutableMap<StateActionKey<*, *>, Int> = mutableMapOf()

    /**
     * Estimates a new Q-function based on the given trajectory and the current Q-function using
     * the On-Policy Monte Carlo method. The function iteratively updates action-value estimates
     * for state-action pairs in the trajectory by incorporating the cumulative return (`G`).
     *
     * If the `firstVisitOnly` flag is enabled, only the first occurrence of each state-action pair
     * in the trajectory is considered for updates.
     *
     * @param Q the current Q-function representing estimated action values for state-action pairs.
     * @param trajectory the sequence of state-action-reward transitions, representing an episode
     *        experienced by the agent.
     * @return the updated Q-function incorporating the returns observed in the trajectory.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
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
