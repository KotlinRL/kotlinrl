package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * A Q-function estimator implementation specifically designed for the SARSA algorithm in reinforcement learning.
 *
 * The SARSAQFunctionEstimator uses the SARSA (State-Action-Reward-State-Action) update rule to modify a Q-function
 * based on observed transitions and their associated temporal differences (TD errors). SARSA performs on-policy
 * learning, meaning that it updates Q-values based on the actual policy being followed rather than an optimal policy.
 *
 * This estimator maintains information about the latest transition encountered, which is used in conjunction with the
 * next action selected by the policy to calculate the TD error and update the Q-function. The update incorporates
 * the learning rate defined by the `alpha` parameter schedule and the discount factor `gamma`, which specifies the
 * importance of future rewards relative to immediate rewards. When the transition marks the end of an episode, the
 * estimator resets its memory of the last transition.
 *
 * Key features:
 * - Supports on-policy reinforcement learning via the SARSA update rule.
 * - Uses a parameterized learning rate schedule (`alpha`).
 * - Accounts for discounting of future rewards with `gamma`.
 * - Resets after terminal states to ensure proper handling of episode boundaries.
 *
 * @param State the type representing the state space of the environment.
 * @param Action the type representing the action space of the environment.
 * @param alpha the learning rate, represented as a [ParameterSchedule], which may be constant or adaptive over time.
 * @param gamma the discount factor for future rewards, ranging between 0 and 1.
 * @param td the temporal-difference error computation strategy, defaulting to SARSA's on-policy update rule.
 */
class SARSAQFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action> = TDQErrors.sarsa()
) : TransitionQFunctionEstimator<State, Action> {
    private var last: Transition<State, Action>? = null

    /**
     * Updates the Q-function based on the given transition using the TD (Temporal Difference) method.
     * The update adjusts the Q-value of the previous state-action pair using the calculated TD error
     * and the provided learning rate. If the previous transition is not available, the method returns
     * the original Q-function without modification.
     *
     * @param Q The current Q-function used for evaluating state-action pairs.
     * @param transition The transition object containing the current state, action, reward, next state,
     * and termination status information for the environment.
     * @return The updated Q-function reflecting the learning step, or the original Q-function if no
     * previous transition is available.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
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