package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements a Q-function updating strategy using the Q-Learning algorithm, based on state-action transitions.
 *
 * This class updates the Q-function by applying the Temporal Difference (TD) learning rule specific to Q-Learning.
 * The TD error is calculated using the difference between the observed reward and the calculated future Q-value.
 * The updated Q-value for a state-action pair minimizes this TD error, allowing the Q-function to better approximate
 * the true state-action values. The update process considers a learning rate schedule (`alpha`) and a discount factor (`gamma`).
 *
 * The class operates in an off-policy learning context, meaning it updates the Q-function based on the optimal
 * action (maximum Q-value) in the subsequent state, regardless of the current policy's action choice.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions performable in the environment.
 * @param alpha a parameter schedule defining the learning rate for updating the Q-function.
 * @param gamma the discount factor, a value between 0 and 1, weighing the importance of future rewards.
 * @param td the temporal difference error function used for Q-value updates, defaulting to Q-Learning.
 */
class QLearningEstimateQ_fromTransition<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action> = TDQErrors.qLearning()
) : EstimateQ_fromTransition<State, Action> {
    /**
     * Updates the Q-function using a state-action transition and the Q-Learning algorithm.
     *
     * This method applies the Temporal Difference (TD) learning rule to adjust the Q-value
     * of the given state-action pair based on the observed transition. If the TD error (`delta`)
     * is zero, the Q-function remains unchanged. Otherwise, the Q-value is updated using the
     * specified learning rate (`alpha`) and the computed TD error adjusted by the current Q-value.
     *
     * @param Q the Q-function storing the Q-values for state-action pairs.
     * @param transition the state-action transition containing the information about
     * the current state, executed action, reward, and next state.
     * @return the updated Q-function with the adjusted Q-value for the given state-action pair.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        transition: Transition<State, Action>
    ): QFunction<State, Action> {
        val (s, a) = transition
        val delta = td(Q, transition, null, gamma, transition.done)
        if (delta == 0.0) return Q
        return Q.update(s, a, Q[s, a] + alpha() * (delta - Q[s, a]))
    }
}
