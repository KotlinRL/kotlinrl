package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the Expected SARSA Q-function estimator for reinforcement learning. This class is
 * responsible for updating an existing Q-function based on the Expected SARSA temporal difference (TD)
 * error, which incorporates the expected value of the next state-action pair under the current policy.
 *
 * Expected SARSA improves stability over traditional SARSA by using the expectation over all possible
 * actions weighted by their policy probabilities, rather than a single sampled action. It balances
 * characteristics between fully on-policy (SARSA) and fully off-policy (Q-Learning) methods.
 *
 * The estimator dynamically reflects policy changes by recomputing the TD error function whenever the policy
 * is updated. This design ensures consistency between the policy and the expected TD error.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial Q-function-based policy used for deriving probabilities and action selection.
 * @param alpha a [ParameterSchedule] representing the step size or learning rate, which can adapt over time.
 * @param gamma the discount factor for weighting future rewards, constrained to a value between 0 and 1.
 * @param initialTD the method for calculating temporal difference (TD) error, defaulting to
 * the Expected SARSA method based on the provided initial policy.
 */
class ExpectedSARSAQFunctionEstimator<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    initialTD: TDQError<State, Action> = TDQErrors.expectedSarsa(initialPolicy)
) : TransitionQFunctionEstimator<State, Action> {
    private var td: TDQError<State, Action> = initialTD

    var policy = initialPolicy
        set(value) {
            field = value
            td = TDQErrors.expectedSarsa(field)
        }

    /**
     * Estimates and updates the Q-function for a given state-action transition using the Expected SARSA method.
     * The estimation process adjusts the Q-value based on the temporal difference (TD) error computed from the
     * current Q-function and the provided transition, incorporating learning rate and discount factors.
     *
     * @param Q the Q-function to be updated, representing the approximation of the optimal action-value function.
     * @param transition the state-action transition that provides the current state, action, reward, next state,
     * and a flag indicating if the episode is done.
     * @return the updated Q-function after incorporating the TD error for the given transition.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val done = transition.done
        val delta = td(Q, transition, null, gamma, done)
        if (delta == 0.0) return Q
        return Q.update(s, a, Q[s, a] + alpha() * delta)
    }
}