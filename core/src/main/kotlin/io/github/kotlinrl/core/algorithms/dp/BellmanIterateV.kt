package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Implements an iterative policy improvement algorithm using the Bellman equations
 * to update value and Q-functions for a given Markov Decision Process (MDP). The
 * class refines the policy through repeated Bellman backups of both value and action-value
 * functions until convergence, as determined by a specified threshold.
 *
 * The algorithm uses a probabilistic trajectory from the MDP model to estimate the updated
 * value function and Q-function while leveraging customizable estimation strategies for both.
 * Once the updates converge, a greedy policy is extracted from the final Q-function.
 *
 * This implementation supports a flexible discount factor (gamma), a convergence threshold
 * (theta), and customizable callbacks for observing value function updates.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions available in the environment.
 * @param gamma the discount factor controlling the importance of immediate versus future rewards. Default is 0.99.
 * @param theta the threshold for determining convergence of the iterative updates. Default is 1e-6.
 * @param stateActions a function that maps states to their corresponding available actions.
 * @param estimateV the strategy for computing value function updates from probabilistic trajectories, defaulting to BellmanEstimateV.
 * @param estimateQ the strategy for computing action-value function (Q-function) updates from probabilistic trajectories, defaulting to BellmanEstimateQ.
 * @param onValueFunctionUpdate a callback invoked on each update of the value function, allowing observation or logging of changes.
 */
class BellmanIterateV<State, Action>(
    private var gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    stateActions: StateActions<State, Action>,
    private val estimateV: EstimateV_fromProbabilisticTrajectory<State, Action> = BellmanEstimateV(gamma),
    private val estimateQ: EstimateQ_fromProbabilisticTrajectory<State, Action> = BellmanEstimateQ(gamma, stateActions),
    private val onValueFunctionUpdate: ValueFunctionUpdate<State> = { },
) : PolicyPlanner<State, Action> {

    /**
     * Executes the Bellman iteration process to estimate an optimal policy based on
     * given Q-function, MDP model, and state-action pairs. This process iteratively
     * refines the value function and Q-function until the change in value function
     * (delta) falls below a predefined threshold (theta).
     *
     * @param Q the initial Q-function representing the current state-action value estimations.
     * @param model the Markov Decision Process (MDP) model providing state transitions and rewards.
     * @param stateActions defines the available actions for each state in the environment.
     * @return a greedy policy derived from the refined Q-function, mapping states to actions
     *         that maximize the Q-values.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        model: MDPModel<State, Action>,
        stateActions: StateActions<State, Action>
    ): Policy<State, Action> {
        var delta: Double
        var currentV = Q.toV()
        var currentQ = Q
        val trajectory = model.probabilisticTrajectory(stateActions)

        do {
            val newV = estimateV(currentV, trajectory)
            val newQ = estimateQ(currentQ, trajectory)

            delta = model.deltaV(stateActions, currentV, newV)

            currentV = newV
            onValueFunctionUpdate(currentV)
            currentQ = newQ
        } while (delta > theta)

        return GreedyPolicy(currentQ, stateActions)
    }
}
