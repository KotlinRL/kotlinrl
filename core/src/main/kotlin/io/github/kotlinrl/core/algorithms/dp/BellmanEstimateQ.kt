package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Represents an implementation of Q-value estimation for a reinforcement learning agent using
 * the Bellman equations. This class is designed to update a Q-function based on a probabilistic
 * trajectory of state-action transitions derived from the environment, leveraging the Bellman
 * backup equations.
 *
 * The BellmanEstimateQ class supports a customizable Bellman backup strategy, including discounted
 * formulations, to calculate expected cumulative rewards for state-action pairs. The update mechanism
 * processes trajectories by grouping transitions by state-action pairs, calculating the expected
 * value for each pair based on observed transitions, and applying the Bellman backup for each update.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param gamma the discount factor used in the Bellman backup calculation. Must be a value in [0, 1].
 * @param stateActions a function that determines possible actions for a given state.
 * @param bellmanBackup a customizable instance of `BellmanBackup` defining the logic for computing
 *        value updates, defaulting to discounted backups using the provided gamma value.
 */
class BellmanEstimateQ<State, Action>(
    private val gamma: Double,
    private val stateActions: StateActions<State, Action>,
    private val bellmanBackup: BellmanBackup<State, Action> = BellmanBackups.discounted(gamma)
) : EstimateQ_fromProbabilisticTrajectory<State, Action> {

    /**
     * Updates the Q-function based on a given probabilistic trajectory using the Bellman backup algorithm.
     *
     * @param Q the current Q-function representing the action-value estimates for state-action pairs.
     * @param trajectory the probabilistic trajectory, which is a sequence of transitions, each specifying
     *                   the state, action, resulting state, reward, probability, and whether it is terminal.
     * @return the updated Q-function after applying the Bellman backup calculations for all state-action pairs
     *         in the provided trajectory.
     */
    override operator fun invoke(
        Q: QFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): QFunction<State, Action> {
        var updatedQ = Q

        val grouped = trajectory.groupBy { it.state to it.action }

        for ((stateAction, transitions) in grouped) {
            val (s, a) = stateAction

            val expectedValue = transitions.sumOf { t ->
                val futureActions = stateActions(t.nextState)
                val maxQ = if (t.done || futureActions.isEmpty()) 0.0
                else futureActions.maxOf { aPrime -> Q[t.nextState, aPrime] }
                bellmanBackup(t.reward, maxQ, t.probability, t.done)
            }

            updatedQ = updatedQ.update(s, a, expectedValue)
        }

        return updatedQ
    }
}
