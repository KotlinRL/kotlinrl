package io.github.kotlinrl.core.algorithms.td.dyna

import io.github.kotlinrl.core.*

/**
 * Implements the Dyna-Q reinforcement learning algorithm, which combines model-based planning
 * with direct reinforcement learning. The algorithm uses an environment model to simulate
 * additional transitions and integrates these simulated experiences with real-world transitions
 * to update both the policy and the Q-function efficiently.
 *
 * This algorithm leverages a learned environment model to perform additional "planning steps"
 * in addition to the direct Q-learning updates, accelerating convergence and improving
 * performance in domains where an accurate model can be learned. The number of planning steps,
 * step size, and discount factor are configurable.
 *
 * The model captures the relationship between states, actions, rewards, and next states
 * and updates itself through observed transitions. These capabilities make Dyna-Q suitable
 * for environments with both deterministic and stochastic dynamics.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be taken in the environment.
 * @param initialPolicy the initial decision-making policy used by the agent.
 * @param alpha a schedule controlling the learning rate of Q-function updates.
 * @param gamma the discount factor, which determines the weight of future rewards.
 * @param model a learnable model of the environment that supports planning by simulating state-action transitions.
 * @param planningSteps the number of simulated planning steps performed for every real-world transition. Defaults to 5.
 * @param estimateQ the function responsible for calculating updates to the Q-function
 * by combining real and simulated experiences. Defaults to the Dyna-Q estimator (`DynaQEstimateQ_fromTransition`).
 * @param onQFunctionUpdate a callback invoked whenever the Q-function is updated.
 * @param onPolicyUpdate a callback invoked whenever the policy is updated.
 */
class DynaQ<State, Action>(
    initialPolicy: Policy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    model: LearnableMDPModel<State, Action>,
    planningSteps: Int = 5,
    estimateQ: EstimateQ_fromTransition<State, Action> = DynaQEstimateQ_fromTransition(alpha, gamma, model, planningSteps),
    onQFunctionUpdate: QFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionLearningAlgorithm<State, Action>(initialPolicy, estimateQ, onPolicyUpdate,onQFunctionUpdate)
