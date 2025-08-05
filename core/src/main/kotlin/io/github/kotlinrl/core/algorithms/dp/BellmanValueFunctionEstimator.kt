package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Implements a Bellman-based value function estimation for discrete Markov Decision Processes (MDPs).
 *
 * This class is used to estimate the value function for a given sequence of states and actions
 * (known as a trajectory) by applying the Bellman expectation update rule. The estimation process
 * updates the state values based on a probabilistic trajectory, taking into account the rewards,
 * state transitions, and a discount factor.
 *
 * @param State the type that represents the states in the Markov Decision Process.
 * @param Action the type that represents the possible actions in the Markov Decision Process.
 * @param gamma the discount factor in the range [0, 1), indicating the importance of future rewards
 *              relative to immediate rewards. The default value is 0.99.
 */
class BellmanValueFunctionEstimator<State, Action>(
    private val gamma: Double = 0.99
) : DPValueFunctionEstimator<State, Action> {

    /**
     * Estimates and updates the value function based on a given probabilistic trajectory using the
     * Bellman expectation update rule.
     *
     * This method calculates and updates the value of each distinct state in the trajectory by computing
     * the expected reward and accounting for future value discounted by the gamma factor.
     *
     * @param V the current enumerable value function that maps states to their estimated scalar values.
     * @param trajectory a probabilistic trajectory containing the sequence of state transitions,
     *                   associated rewards, probabilities, and terminal state indicators.
     * @return an updated enumerable value function with new values for the states in the trajectory.
     */
    override fun estimate(
        V: EnumerableValueFunction<State>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableValueFunction<State> {
        var updatedV = V

        val states = trajectory.map { it.state }.distinct()
        for (s in states) {
            val transitionsFromS = trajectory.filter { it.state == s }

            val newValue = transitionsFromS.sumOf { transition ->
                val r = transition.reward
                val sPrime = transition.nextState
                val p = transition.probability
                val done = transition.done

                val value = if (done) 0.0 else V[sPrime]
                p * (r + gamma * value)
            }

            updatedV = updatedV.update(s, newValue)
        }

        return updatedV
    }
}
