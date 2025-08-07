package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Implements a value function estimation strategy using the Bellman expectation update rule for a given
 * probabilistic trajectory. This class is used in reinforcement learning settings to iteratively refine
 * the value function through backup calculations.
 *
 * The estimation process is guided by a Bellman backup function, which defines the exact formulation
 * for computing value updates. By default, a discounted Bellman backup with the specified discount factor
 * (gamma) is used, enabling updates based on both immediate rewards and the discounted sum of future rewards.
 *
 * This class operates on transitions derived from trajectories, where each transition includes states, actions,
 * rewards, and probabilities. For each state-action pair observed in the trajectory, the associated transitions
 * are aggregated to compute the expected return using the Bellman backup function.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be performed in the environment.
 * @param gamma discount factor for future rewards, controlling the emphasis on immediate versus future rewards.
 * @param bellmanBackup a rule for computing value updates, encapsulating immediate rewards, future rewards,
 *                      and probabilistic transitions using a Bellman backup formulation.
 */
class BellmanEstimateV<State, Action>(
    private val gamma: Double = 0.99,
    private val bellmanBackup: BellmanBackup<State, Action> = BellmanBackups.discounted(gamma)
) : EstimateV_fromProbabilisticTrajectory<State, Action> {

    /**
     * Performs a Bellman update on the given value function using a probabilistic trajectory.
     * This update iterates through all distinct states and their possible actions within the
     * trajectory, calculating the updated value function based on the Bellman equation.
     *
     * @param V the current value function to be updated.
     * @param trajectory a probabilistic trajectory representing the sequence of state-action-reward-nextState
     *                   transitions observed in the Markov Decision Process (MDP).
     * @return the updated value function after performing the Bellman update.
     */
    override operator fun invoke(
        V: ValueFunction<State>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): ValueFunction<State> {
        var updatedV = V

        val states = trajectory.map { it.state }.distinct()
        for (s in states) {
            val actions = trajectory.filter { it.state == s }.map { it.action }.distinct()

            val bestActionValue = actions.maxOf { a ->
                trajectory.filter { it.state == s && it.action == a }.sumOf { t ->
                    val value = if (t.done) 0.0 else V[t.nextState]
                    bellmanBackup(t.reward, value, t.probability, t.done)
                }
            }

            updatedV = updatedV.update(s, bestActionValue)
        }

        return updatedV
    }
}
