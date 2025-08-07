package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.base.EstimateQ_fromProbabilisticTrajectory
import io.github.kotlinrl.core.algorithms.base.PolicyPlanner

/**
 * Implements a policy planner using iterative updates of Q-values based on the Bellman equation until
 * convergence. This class aims to refine the Q-function to approximate optimal action-value estimates
 * and derives a greedy policy for decision-making.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the allowable actions in the environment.
 * @param gamma the discount factor for future rewards during the Bellman backup calculation.
 *              Must be within the range [0, 1].
 * @param theta the convergence threshold for Q-value updates. Iterations stop when the maximum change
 *              in Q-value across all state-action pairs is less than or equal to this value.
 * @param stateActions a function providing possible actions for a given state.
 * @param estimateQ an `EstimateQ_fromProbabilisticTrajectory` instance used for updating Q-values by applying
 *                  the Bellman backup on a probabilistic trajectory. Defaults to `BellmanEstimateQ`.
 * @param onQFunctionUpdate a callback invoked after every Q-function update during the iteration process.
 */
class BellmanIterateQ<State, Action>(
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    private val estimateQ: EstimateQ_fromProbabilisticTrajectory<State, Action> = BellmanEstimateQ(gamma, stateActions),
    private val onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
) : PolicyPlanner<State, Action> {

    /**
     * Iteratively refines the Q-function using the Bellman equation until convergence and derives a
     * greedy policy based on the refined Q-function.
     *
     * @param Q the initial `QFunction` representing the action-value function of the Markov Decision Process (MDP).
     * @param model the `MDPModel` containing information on state transitions, rewards, and environment dynamics.
     * @param stateActions a mapping of states to their possible actions, defining the available actions for each state.
     * @return a `Policy` that acts greedily with respect to the optimized Q-function, selecting actions to maximize the expected reward.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        model: MDPModel<State, Action>,
        stateActions: StateActions<State, Action>
    ): Policy<State, Action> {
        var delta: Double
        var currentQ = Q
        val trajectory = model.probabilisticTrajectory(stateActions)

        do {
            val newQ = estimateQ(currentQ, trajectory)
            delta = model.deltaQ(stateActions, currentQ, newQ)
            currentQ = newQ
            onQFunctionUpdate(currentQ)
        } while (delta > theta)

        return GreedyPolicy(currentQ, stateActions)
    }
}
