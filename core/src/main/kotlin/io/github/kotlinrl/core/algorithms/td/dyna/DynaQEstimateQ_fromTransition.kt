package io.github.kotlinrl.core.algorithms.td.dyna

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.td.TDQErrors

/**
 * Implements the Dyna-Q reinforcement learning algorithm for updating a Q-function
 * based on real experiences and simulated transitions from a learned model of the environment.
 *
 * The Dyna-Q algorithm combines direct reinforcement learning updates (based on real transitions)
 * with a planning phase where simulated transitions are used for additional updates. This approach
 * is particularly useful when computational resources allow for extra updates via planning, improving
 * learning efficiency.
 *
 * @param State the type representing the environment's states.
 * @param Action the type representing the actions that can be taken in the environment.
 * @param alpha the parameter schedule controlling the learning rate for Q-function updates.
 * @param gamma the discount factor used to determine the importance of future rewards, ranging from 0 to 1.
 * @param model the learnable model of the environment, which maintains and updates knowledge about
 * the state transitions and rewards.
 * @param planningSteps the number of planning updates to perform using simulated transitions during each step.
 * Defaults to 5.
 * @param td the temporal difference (TD) error functionality to use for Q-value updates. By default, it uses
 * the Q-Learning algorithm.
 */
class DynaQEstimateQ_fromTransition<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val model: LearnableMDPModel<State, Action>,
    private val planningSteps: Int = 5,
    private val td: TDQError<State, Action> = TDQErrors.qLearning()
) : EstimateQ_fromTransition<State, Action> {

    /**
     * Invokes the function to estimate and update the Q-function based on the given transition and planning steps.
     * This method combines real experience learning with model-based planning to improve the Q-function.
     *
     * @param Q the current Q-function that maps state-action pairs to their estimated values.
     * @param transition the actual transition observed, consisting of the current state, action taken, reward obtained,
     * next state, and a flag indicating whether the episode has ended.
     * @return the updated Q-function after applying both the real experience update and the model-based planning updates.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        transition: Transition<State, Action>
    ): QFunction<State, Action> {
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