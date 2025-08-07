package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Implements a hybrid policy planning algorithm that integrates both model-based
 * and model-free reinforcement learning techniques. The class uses policy planning
 * to derive updated policies and Q-function estimates based on the learning model
 * and environment interactions. This allows for dynamic adaptation of the policy
 * based on observed transitions and trajectories, leveraging both local updates
 * and trajectory-level refinements.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the possible actions within the environment.
 * @param initialPolicy the initial policy providing the starting point for policy planning.
 * @param model the Markov Decision Process (MDP) model used to predict state transitions and rewards.
 * @param stateActions mappings of states to their available actions within the environment.
 * @param policyPlanner a strategy for generating policies from the Q-function, MDP model, and state-action mappings.
 * @param estimateQ_fromTransition a function for updating the Q-function based on observed transitions.
 *        Defaults to null if no transition-based estimator is provided.
 * @param estimateQ_fromTrajectory a function for updating the Q-function from a trajectory of state-action-reward transitions.
 *        Defaults to null if no trajectory-based estimator is provided.
 * @param onPolicyUpdate an optional callback function that triggers whenever the policy is recalculated.
 *        Defaults to an empty callback function.
 * @param onQFunctionUpdate an optional callback function that triggers whenever the Q-function is updated.
 *        Defaults to an empty callback function.
 */
class HybridPolicyPlanningAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val model: MDPModel<State, Action>,
    private val stateActions: StateActions<State, Action>,
    private val policyPlanner: PolicyPlanner<State, Action>,
    private val estimateQ_fromTransition: EstimateQ_fromTransition<State, Action>? = null,
    private val estimateQ_fromTrajectory: EstimateQ_fromTrajectory<State, Action>? = null,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
) : BaseAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    onPolicyUpdate = onPolicyUpdate,
    onQFunctionUpdate = onQFunctionUpdate
) {

    init {
        policy = policyPlanner(Q, model, stateActions)
    }

    /**
     * Updates the hybrid policy planning algorithm with the provided state-action transition.
     *
     * This method incorporates the given transition into the model's knowledge representation
     * and updates the Q-function using a transition-based estimator. It then re-plans the policy
     * based on the updated Q-function, model, and state-action mappings. If the model is not
     * learnable or if the Q-function transition estimator is unavailable, this method performs
     * no operation.
     *
     * @param transition The state-action-reward-next-state transition used to update the model, Q-function, and policy.
     */
    override fun update(transition: Transition<State, Action>) {
        if (model !is LearnableMDPModel || estimateQ_fromTransition == null) return

        model.update(transition)
        Q = estimateQ_fromTransition(Q, transition)
        policy = policyPlanner(Q, model, stateActions)
    }

    /**
     * Updates the hybrid policy planning algorithm using the provided trajectory of state-action transitions
     * and the current episode number. This method utilizes the trajectory to update the model's internal
     * state, refines the Q-function using a trajectory-based estimator, and re-plans the policy with the
     * updated Q-function, model, and state-action mappings.
     *
     * If the model is not learnable or the trajectory-based Q-function estimator is unavailable, the method
     * performs no operation.
     *
     * @param trajectory the sequence of state-action-reward transitions representing an episode within
     *        the environment.
     * @param episode the index of the current episode during the learning process.
     */
    override fun update(trajectory: Trajectory<State, Action>, episode: Int) {
        if (model !is LearnableMDPModel || estimateQ_fromTrajectory == null) return

        trajectory.forEach(model::update)
        Q = estimateQ_fromTrajectory(Q, trajectory)
        policy = policyPlanner(Q, model, stateActions)
    }
}