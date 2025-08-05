package io.github.kotlinrl.core.algorithms.td.dyna

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.td.TDQErrors

/**
 * Implements the Dyna-Q estimator for updating Q-functions in reinforcement learning.
 * This estimator combines real-world experience with simulated planning steps to improve
 * the learning process, updating the Q-function using both direct transitions and additional
 * transitions simulated by a learned model of the environment.
 *
 * The real-world experience is incorporated using Q-learning updates, and the model is updated
 * based on observed transitions. Then, the model is used to generate simulated transitions
 * that further refine the Q-function through additional updates. This approach accelerates
 * learning by leveraging a combination of direct and simulated experiences.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be taken in the environment.
 * @param alpha a schedule that provides the step size for updates to the Q-function.
 * Controls the learning rate of the algorithm.
 * @param gamma the discount factor that determines the importance of future rewards
 * in the Q-function updates.
 * @param model a learnable model of the environment that captures state-action transitions
 * and their associated rewards. This model is updated with real experience and used for planning steps.
 * @param planningSteps the number of simulated planning steps per real-world transition.
 * Defaults to 5.
 * @param td the temporal difference error function used to compute the update target for Q-values.
 * Defaults to Q-learning as the temporal difference update mechanism.
 */
class DynaQEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val model: LearnableMDPModel<State, Action>,
    private val planningSteps: Int = 5,
    private val td: TDQError<State, Action> = TDQErrors.qLearning()
) : TransitionQFunctionEstimator<State, Action> {

    /**
     * Updates the Q-function using a combination of real experience and simulated planning transitions
     * based on the Dyna-Q reinforcement learning algorithm. The method performs the following steps:
     * - Updates the Q-function using the real experience from the given state transition.
     * - Updates the internal model of the environment with the observed transition.
     * - Executes planning by sampling transitions from the model to further update the Q-function.
     *
     * @param Q the current Q-function, representing the state-action value estimates.
     * @param transition the real environment state transition, represented by the current state, action,
     * next state, reward, and whether the transition is terminal.
     * @return the updated Q-function after incorporating both real experience and planning steps.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val done = transition.done

        // Real experience Q-learning
        val delta = td(Q, transition, null, gamma, done)
        var currentQ = Q.update(s, a, Q[s, a] + alpha() * delta)

        // Update model
        model.update(transition)

        // Planning phase
        repeat(planningSteps) {
            val sampleTransition = model.sampleTransition() ?: return@repeat
            val (sPlan, aPlan) = sampleTransition
            val done = sampleTransition.done
            val deltaPlan = td(currentQ, sampleTransition, null, gamma, done)
            currentQ = currentQ.update(sPlan, aPlan, currentQ[sPlan, aPlan] + alpha() * deltaPlan)
        }
        return currentQ
    }
}