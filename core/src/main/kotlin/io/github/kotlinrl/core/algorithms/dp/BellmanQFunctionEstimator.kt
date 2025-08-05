package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * An implementation of a Q-function estimator using the Bellman update equation
 * for Markov Decision Processes (MDPs). This estimator applies the Bellman expectation equation
 * to update Q-values based on observed trajectories in the environment.
 *
 * The primary responsibility of this class is to estimate an updated Q-function
 * given an existing Q-function and a probabilistic trajectory of state-action transitions.
 * Updates are performed by calculating the expected cumulative reward for each state-action pair.
 *
 * @param State the type representing the states in the MDP.
 * @param Action the type representing the actions in the MDP.
 * @param gamma the discount factor in the range [0, 1), which determines the importance of future rewards.
 * @param stateActions a function that determines the list of possible actions for a given state.
 */
class BellmanQFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val stateActions: StateActions<State, Action>
) : DPQFunctionEstimator<State, Action> {

    /**
     * Estimates an updated Q-function using a probabilistic trajectory of state-action transitions.
     * This method applies the Bellman update equation to compute new Q-values for each state-action
     * pair present in the given trajectory, taking into account the expected future rewards.
     *
     * @param Q the initial Q-function representing the current estimates of Q-values for state-action pairs.
     * @param trajectory a probabilistic trajectory detailing the observed transitions, including
     *                   probabilities, rewards, and next states for each state-action pair.
     * @return an updated Q-function that incorporates the calculated expected values based on the trajectory.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        var updatedQ = Q

        val grouped = trajectory.groupBy { it.state to it.action }

        for ((stateAction, transitions) in grouped) {
            val (s, a) = stateAction

            val expectedValue = transitions.sumOf { t ->
                val futureActions = stateActions(t.nextState)
                val maxQ = if (t.done || futureActions.isEmpty()) 0.0
                else futureActions.maxOf { aPrime -> Q[t.nextState, aPrime] }

                t.probability * (t.reward + gamma * maxQ)
            }

            updatedQ = updatedQ.update(s, a, expectedValue)
        }

        return updatedQ
    }
}
