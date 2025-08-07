package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the SARSA algorithm for updating a Q-function based on transitions in a reinforcement
 * learning environment.
 *
 * SARSA (State-Action-Reward-State-Action) is an on-policy Temporal Difference (TD) control
 * algorithm. It updates Q-value estimates for state-action pairs by evaluating the TD error
 * computed using the reward from the transition and the Q-value of the next state-action pair
 * under the current policy. This implementation uses a parameter schedule for the learning rate
 * (alpha) and supports customizable TD error calculations.
 *
 * This class maintains the last encountered transition to ensure updates are based on consecutive
 * transitions. If no previous transition is available, the Q-function remains unaltered.
 *
 * @param State The type representing the states of the environment.
 * @param Action The type representing the actions performable in the environment.
 * @param alpha The learning rate schedule used to determine the step size for updates.
 * @param gamma The discount factor, a value in [0, 1], which adjusts the importance
 * of future rewards relative to immediate rewards.
 * @param td The Temporal Difference (TD) error calculation strategy, defaulting to the SARSA method.
 */
class SARSAEstimateQ_fromTransition<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action> = TDQErrors.sarsa()
) : EstimateQ_fromTransition<State, Action> {
    private var last: Transition<State, Action>? = null

    /**
     * Updates the Q-function based on a given transition and returns the updated Q-function.
     *
     * This method implements the SARSA (State-Action-Reward-State-Action) update logic, an on-policy
     * Temporal Difference (TD) learning method. By leveraging the previous and current transitions,
     * it calculates the TD error and uses it to update the Q-value associated with a specific
     * state-action pair. The update is scaled by a dynamically computed learning rate (alpha).
     *
     * The method also handles terminal states by resetting the previous transition information if
     * the current transition marks the end of an episode.
     *
     * @param Q the current Q-function, used to retrieve and update Q-values for state-action pairs.
     * @param transition the most recent transition, containing information about the current state,
     * action, reward, next state, and whether the episode has ended.
     * @return the updated Q-function after applying the SARSA update rule.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        transition: Transition<State, Action>
    ): QFunction<State, Action> {
        val prev = last
        last = transition

        if (prev == null) return Q

        val (s, a) = prev
        val (_, aPrime) = transition
        val delta = td(Q, prev, aPrime, gamma, transition.done)
        if (delta == 0.0) return Q
        if (transition.done) last = null

        return Q.update(s, a, Q[s, a] + alpha() * (delta - Q[s, a]))
    }
}