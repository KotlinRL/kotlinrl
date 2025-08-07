package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * A class for estimating and updating Q-values using the Expected SARSA algorithm and a given policy.
 * This implementation adapts Q-values by considering the expected next-state value, factoring in
 * probabilistic action selection as defined by the policy. It is designed for reinforcement learning
 * tasks where the policy and environment dynamics may evolve over time.
 *
 * @param State the type representing the states of the environment.
 * @param Action the type representing the actions performable in the environment.
 * @param initialPolicy the initial policy used to calculate expected Q-values for updates,
 * which can later be adjusted to adapt to changing strategies.
 * @param alpha a parameter schedule controlling the learning rate during updates, enabling
 * dynamic adjustment of the step size over time or iterations.
 * @param gamma the discount factor determining the weight of future rewards, a value typically
 * between 0 and 1. A higher gamma assigns greater significance to long-term rewards, whereas a
 * lower gamma emphasizes short-term gains.
 * @param initialTD the initial temporal difference (TD) error calculation method, defaulting
 * to the Expected SARSA TD error computed based on the provided initial policy.
 */
class ExpectedSARSAQEstimateQ_fromTransition<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    initialTD: TDQError<State, Action> = TDQErrors.expectedSarsa(initialPolicy)
) : EstimateQ_fromTransition<State, Action> {
    private var td: TDQError<State, Action> = initialTD

    /**
     * Represents the policy being used for reinforcement learning, which determines the selection
     * of actions in various states based on the associated Q-function. The policy is dynamically
     * adjustable and influences the computation of the temporal difference (TD) error for updating
     * the Q-function.
     *
     * When the policy is updated, the TD error computation is automatically refreshed using the
     * `expectedSarsa` method with the new policy. This ensures synchronization between the policy
     * and the TD error calculations, maintaining consistency in the learning process.
     */
    var policy = initialPolicy
        set(value) {
            field = value
            td = TDQErrors.expectedSarsa(field)
        }

    /**
     * Updates the given Q-function by applying the Expected SARSA temporal difference
     * learning rule using the provided transition.
     *
     * The method computes the temporal difference (TD) error based on the current
     * state-action pair, observed reward, and the expected next state-action values.
     * If the TD error is zero, the Q-function is returned unchanged. Otherwise, the Q-function
     * is updated by applying a learning rate to the TD error, and the adjusted Q-value is stored.
     *
     * @param Q the Q-function to be updated, representing the quality of state-action pairs.
     * @param transition the transition representing the current state, action, reward,
     * next state, and whether the episode has terminated.
     * @return the updated Q-function after applying the Expected SARSA update rule.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        transition: Transition<State, Action>
    ): QFunction<State, Action> {
        val (s, a) = transition
        val done = transition.done
        val delta = td(Q, transition, null, gamma, done)
        if (delta == 0.0) return Q
        return Q.update(s, a, Q[s, a] + alpha() * delta)
    }
}