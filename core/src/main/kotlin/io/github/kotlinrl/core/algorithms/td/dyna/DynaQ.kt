package io.github.kotlinrl.core.algorithms.td.dyna

import io.github.kotlinrl.core.*

/**
 * Implements the Dyna-Q reinforcement learning algorithm, combining real experience
 * with simulated planning to improve the learning process. The algorithm updates
 * the Q-function using both direct experiences and simulated experiences generated
 * from a learnable MDP model.
 *
 * The Dyna-Q algorithm works by observing environment transitions, updating a model
 * of the environment, and using the model to simulate additional transitions (planning
 * steps) to further refine the Q-function. This dual approach of real and simulated
 * sample updates accelerates learning compared to pure Q-learning.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the possible actions in the environment.
 * @param initialPolicy the initial Q-function-based policy used by the algorithm.
 * This policy determines the agent's behavior before Q-function improvement.
 * @param alpha the step size schedule for updating Q-values, influencing the learning rate.
 * @param gamma the discount factor, determining the significance of future rewards.
 * @param model the environment model capable of learning and simulating
 * state-action transitions. It is essential for planning.
 * @param planningSteps the number of simulated transitions to generate per step
 * for planning. Defaults to 5.
 * @param estimator the Q-function estimator used for transition-based updates.
 * Defaults to a DynaQEstimator.
 * @param onQFunctionUpdate a callback function invoked whenever the Q-function
 * is updated.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated.
 */
class DynaQ<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    model: LearnableMDPModel<State, Action>,
    planningSteps: Int = 5,
    estimator: TransitionQFunctionEstimator<State, Action> = DynaQEstimator(alpha, gamma, model, planningSteps),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TransitionQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate,onQFunctionUpdate)
