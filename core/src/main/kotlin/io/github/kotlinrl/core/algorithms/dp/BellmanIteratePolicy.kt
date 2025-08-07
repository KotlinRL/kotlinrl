package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Implements the Bellman policy iteration algorithm for finding the optimal policy
 * in a Markov Decision Process (MDP). This class alternates between policy evaluation
 * and policy improvement steps using provided estimators to update value and Q-functions.
 *
 * The policy iteration process involves:
 * - Using the current policy to estimate the value function iteratively until convergence.
 * - Improving the policy by greedily selecting the optimal actions based on the updated
 *   Q-function.
 * - Repeating the evaluation and improvement steps until the policy stabilizes.
 *
 * The algorithm leverages the Bellman equations and probabilistic trajectories to compute
 * value and Q-function updates, ensuring convergence towards an optimal policy.
 *
 * @param State the type representing states in the MDP.
 * @param Action the type representing actions in the MDP.
 * @param initialPolicy the initial policy to start the iteration process.
 * @param gamma the discount factor for future rewards, controlling the weighting of
 *              immediate versus future rewards.
 * @param theta the convergence threshold for value updates. Iteration stops when
 *              the maximum change across states is below this value.
 * @param stateActions a mapping of states to their allowable actions in the MDP.
 * @param estimateV an implementation of `EstimateV_fromProbabilisticTrajectory`, used to
 *                  compute value function updates during policy evaluation.
 * @param estimateQ an implementation of `EstimateQ_fromProbabilisticTrajectory`, used to
 *                  compute Q-value updates during policy improvement.
 * @param onValueFunctionUpdate a callback triggered with the updated value function after
 *                              each iteration of policy evaluation.
 */
class BellmanIteratePolicy<State, Action>(
    private val initialPolicy: Policy<State, Action>,
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    private val estimateV: EstimateV_fromProbabilisticTrajectory<State, Action> = BellmanEstimateV(gamma),
    private val estimateQ: EstimateQ_fromProbabilisticTrajectory<State, Action> = BellmanEstimateQ(gamma, stateActions),
    private val onValueFunctionUpdate: ValueFunctionUpdate<State> = { },
) : PolicyPlanner<State, Action> {

    /**
     * Performs the policy iteration algorithm to compute the optimal policy for a given
     * Markov Decision Process (MDP). The method iteratively refines the policy and value
     * functions until the policy becomes stable, resulting in an optimal policy.
     *
     * @param Q the initial Q-function representing state-action value estimates.
     * @param model the MDP model that defines the states, transitions, rewards, and dynamics.
     * @param stateActions a mapping of states to their possible actions in the MDP.
     * @return the optimal policy derived from the given Q-function and MDP model.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        model: MDPModel<State, Action>,
        stateActions: StateActions<State, Action>
    ): Policy<State, Action> {
        var currentV = Q.toV()
        var currentPolicy = initialPolicy
        var stable: Boolean

        do {
            var delta: Double
            val trajectory = model.probabilisticTrajectory(currentPolicy)
            do {
                val newV = estimateV(currentV, trajectory)
                delta = model.deltaV(stateActions, currentV, newV)
                currentV = newV
                onValueFunctionUpdate(currentV)
            } while (delta > theta)

            stable = true
            val newQ = estimateQ(currentPolicy.Q, trajectory)
            val newPolicy = GreedyPolicy(newQ, stateActions)

            for (s in model.allStates()) {
                if (currentPolicy(s) != newPolicy(s)) {
                    stable = false
                    break
                }
            }

            currentPolicy = newPolicy
        } while (!stable)

        return currentPolicy
    }
}