package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements an on-policy Monte Carlo method to estimate the state-value function (V)
 * from a given trajectory in a reinforcement learning context.
 *
 * This class employs the Monte Carlo estimation technique for updating state values
 * based on observed returns from sampled trajectories. It provides the option to
 * perform either first-visit or every-visit Monte Carlo updates, depending on the value
 * of the `firstVisitOnly` parameter.
 *
 * The returns for each state are computed based on the cumulative rewards observed
 * along the trajectory, with a discount factor (`gamma`) applied to account for the effect
 * of future rewards. These returns are then averaged over multiple visits (depending on the
 * chosen strategy) to produce an updated estimate for the state-value function.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be taken in the environment.
 * @param gamma the discount factor that determines the weight of future rewards.
 * @param firstVisitOnly indicates whether to use only the first visit to each state
 *        in the trajectory for updating the state-value function. Defaults to true.
 */
class OnPolicyMonteCarloEstimateV_fromTrajectory<State, Action>(
    private val gamma: Double,
    private val firstVisitOnly: Boolean = true,
) : EstimateV_fromTrajectory<State, Action> {

    private val returnsCount: MutableMap<Comparable<*>, Int> = mutableMapOf()
    private val returnsSum: MutableMap<Comparable<*>, Double> = mutableMapOf()

    /**
     * Updates the value function based on a given trajectory of states, actions, and rewards
     * using the Monte Carlo method. Depending on the `firstVisitOnly` flag, the method considers
     * either the first visit only or all visits of each state in the trajectory.
     *
     * The method calculates the cumulative discounted reward (return) for each state in the trajectory
     * in reverse order and updates the value function accordingly. This update is based on the average
     * return for each state, computed incrementally using sums and counts stored in the class.
     *
     * @param V the initial value function to be updated. Holds the current value estimates
     *        for each state in the environment.
     * @param trajectory a trajectory of states, actions, and rewards observed during an episode,
     *        which is used to compute the returns and update the value estimates.
     * @return the updated value function containing improved estimates for the states encountered
     *         in the given trajectory.
     */
    override fun invoke(V: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State> {
        val visited = mutableSetOf<Comparable<*>>()
        var G = 0.0
        var updatedV = V

        for ((s, _, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = s.toComparable()

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val count = returnsCount.getOrDefault(key, 0) + 1
            val sum = returnsSum.getOrDefault(key, 0.0) + G

            returnsCount[key] = count
            returnsSum[key] = sum

            val averageReturn = sum / count
            updatedV = updatedV.update(s, averageReturn)
        }

        return updatedV
    }
}
