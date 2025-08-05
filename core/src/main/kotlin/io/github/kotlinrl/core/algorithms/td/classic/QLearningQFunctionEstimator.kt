package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements a Q-Function Estimator specifically for the Q-Learning algorithm.
 *
 * Q-Learning is an off-policy Temporal Difference (TD) learning method where the Q-function is updated
 * using the maximum expected reward from the next state, rather than taking into account the action
 * suggested by the current policy. This estimator calculates and applies the Temporal Difference error
 * (TD error) to update the Q-function.
 *
 * This class uses a learning rate schedule ([alpha]) to control the step size during updates, and a
 * discount factor ([gamma]) to weigh the importance of future rewards. By default, it utilizes the
 * Q-Learning TD error calculation via [TDQErrors.qLearning].
 *
 * The estimator updates the Q-function using the following formula:
 *
 * Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) − Q(s, a)]
 *
 * where:
 * - s: current state
 * - a: current action
 * - r: reward received after taking action a in state s
 * - s': next state
 * - a': next action
 * - α: learning rate
 * - γ: discount factor
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param alpha a schedule for the learning rate used to control the step size during Q-function updates.
 * @param gamma the discount factor applied to future rewards, with a value between 0 and 1.
 * @param td the Temporal Difference error calculation used for weight adjustments, defaulting
 *           to the Q-Learning TD error ([TDQErrors.qLearning]).
 */
class QLearningQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action> = TDQErrors.qLearning()
) : TransitionQFunctionEstimator<State, Action> {
    /**
     * Updates the Q-function based on the given state-action transition using the Temporal Difference (TD) error.
     * If the TD error for the transition is zero, the original Q-function is returned unchanged.
     * Otherwise, the Q-value for the state-action pair is adjusted to reduce the TD error.
     *
     * @param Q the current Q-function to be updated, representing the estimated value of state-action pairs.
     * @param transition the state-action transition to process, consisting of the current state and action taken.
     * @return the updated Q-function after applying the TD update for the given transition.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val delta = td(Q, transition, null, gamma, transition.done)
        if (delta == 0.0) return Q
        return Q.update(s, a, Q[s, a] + alpha() * (delta - Q[s, a]))
    }
}
