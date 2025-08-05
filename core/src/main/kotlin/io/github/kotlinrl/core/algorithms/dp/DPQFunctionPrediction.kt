package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Represents a mechanism to iteratively predict and refine a Q-function in the context of
 * reinforcement learning using Dynamic Programming (DP) techniques.
 *
 * This class combines a Q-function estimator, a Markov Decision Process (MDP) model, and
 * a policy to dynamically update the Q-function. The updates are performed using trajectories
 * generated from transitions in the MDP, which are influenced by the given policy. The refined
 * Q-function serves as an improved estimate of the action-value function.
 *
 * @param State The type representing states in the environment or MDP.
 * @param Action The type representing possible actions that can be performed in the environment or MDP.
 * @constructor Initializes the Q-function predictor with a starting Q-function, a Q-function estimator,
 * and an MDP model. The initial Q-function serves as the base for subsequent refinement processes.
 *
 * @property Q The current estimate of the Q-function, which maps state-action pairs to their
 * respective action-value estimates. This property is updated during the prediction process.
 * It is initialized with the provided `initialQ` and subsequently refined through iterations.
 */
class DPQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: DPQFunctionEstimator<State, Action>,
    private val model: MDPModel<State, Action>,
) {
    /**
     * Represents the current action-value function (Q-function) in the Dynamic Programming
     * (DP) iteration process used for reinforcement learning. This variable maps state-action
     * pairs to their respective estimated values, representing the action-value of taking
     * specific actions in given states.
     *
     * The `Q` function is initialized with the provided `initialQ` and is iteratively
     * updated during each DP iteration using the associated `DPQFunctionEstimator`.
     * Updates to the Q-function are based on transitions generated from the Markov Decision
     * Process (MDP) model influenced by the given policy. The refined Q-function provides
     * progressively better approximations to the optimal action-value function of the MDP.
     *
     * This property can only be externally accessed, while updates are restricted to internal use
     * by the class's methods.
     */
    var Q: EnumerableQFunction<State, Action> = initialQ
        private set

    /**
     * Performs a dynamic programming update to refine the Q-function of the model
     * based on the given policy. This method computes updated action-value estimates
     * by generating a trajectory of state-action transitions and applying the Q-function
     * estimator.
     *
     * @param policy The policy used to determine the actions for each state in the model.
     *               It maps each state to an action that should be taken.
     * @return The updated Q-function representing the estimated action values
     *         for state-action pairs after applying the dynamic programming update.
     */
    operator fun invoke(policy: Policy<State, Action>): EnumerableQFunction<State, Action> {
        val trajectory = model.allStates().flatMap { s ->
            model.transitions(s, policy(s))
        }

        Q = estimator.estimate(Q, trajectory)
        return Q
    }
}
