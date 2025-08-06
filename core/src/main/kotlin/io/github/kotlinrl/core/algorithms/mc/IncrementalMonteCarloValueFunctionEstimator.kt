package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements an incremental Monte Carlo value function estimator.
 *
 * This estimator updates a value function based on a given trajectory of state-action pairs and rewards.
 * Using either first-visit or every-visit Monte Carlo methods, it computes the discounted return (G)
 * for each state encountered in the trajectory and adjusts the value function incrementally using a
 * specified learning rate (alpha). The process iteratively refines the value function to better estimate
 * the expected cumulative reward for each state under the policy that generated the trajectory.
 *
 * @param State the type representing the states of the environment.
 * @param Action the type representing the possible actions in the environment.
 * @param gamma the discount factor, which determines the weight of future rewards when calculating returns.
 * It must be a value from 0.0 (only immediate rewards) to 1.0 (consider all future rewards equally).
 * @param alpha a parameter schedule defining the learning rate used to incrementally update the value function.
 * This learning rate specifies how much the estimator adjusts the current estimate when incorporating
 * new information from a trajectory.
 * @param firstVisitOnly a boolean flag indicating whether to use the first-visit Monte Carlo method
 * (true) or the every-visit Monte Carlo method (false). In first-visit Monte Carlo, only the first
 * occurrence of a state in a trajectory is used to update the value function, while in every-visit,
 * all occurrences are used.
 */
class IncrementalMonteCarloValueFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    private val firstVisitOnly: Boolean = true,
) : TrajectoryValueFunctionEstimator<State, Action> {

    override fun estimate(V: EnumerableValueFunction<State>, trajectory: Trajectory<State, Action>): EnumerableValueFunction<State> {
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
