package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*

/**
 * Implements the TD(λ) (Temporal Difference with Lambda) algorithm for estimating Q-values
 * based on state-action transitions in a reinforcement learning environment.
 *
 * This class operates on the principle of eligibility traces, which allow credit to be distributed
 * across multiple state-action pairs. By combining lambda return methods with a dynamic learning
 * rate (alpha), this implementation effectively updates the Q-function and improves the policy
 * iteratively based on observed transitions.
 *
 * @param State the type representing states in the environment.
 * @param Action the type representing actions that can be performed in the environment.
 * @param initialPolicy the initial policy used for action selection in the learning process.
 * @param alpha the schedule for the learning rate, determining the step size for updates to the Q-function.
 * @param lambda the schedule for the decay factor, controlling the eligibility trace's contribution over time.
 * @param gamma the discount factor, used to weigh the importance of future rewards in updates.
 * @param td the temporal difference error function used to compute the difference between predicted and target Q-values.
 * @param initialEligibilityTrace the initial state of the eligibility trace that determines credit assignment.
 * @param onEligibilityTraceUpdate a callback function that is invoked whenever the eligibility trace is updated.
 */
class TDLambdaEstimateQ_fromTransition<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val alpha: ParameterSchedule,
    private val lambda: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action>,
    initialEligibilityTrace: EligibilityTrace<State, Action>,
    private val onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { }
) : EstimateQ_fromTransition<State, Action> {

    /**
     * Holds the current policy, which maps states to actions or action probabilities.
     *
     * This variable is used to guide action selection during the reinforcement learning
     * process and is updated iteratively as the algorithm improves the policy based on
     * the learned Q-values.
     */
    private var policy: Policy<State, Action> = initialPolicy
    /**
     * Represents the eligibility trace mechanism used in the TD(λ) algorithm for credit assignment to
     * state-action pairs over multiple time steps.
     *
     * This variable is initialized with a default trace mechanism and can be updated dynamically through
     * the setter. Any updates to the trace invoke the callback `onEligibilityTraceUpdate`, allowing
     * additional processing or monitoring on eligibility trace changes.
     *
     * @property trace The current eligibility trace employed by the algorithm. It is updated when
     * necessary to reflect changes in the learning dynamics.
     */
    private var trace: EligibilityTrace<State, Action> = initialEligibilityTrace
        set(value) {
            field = value
            onEligibilityTraceUpdate(value)
        }

    /**
     * Updates the Q-function based on a given transition and adjusts the eligibility trace and policy.
     *
     * This function performs a temporal difference (TD) update on the Q-function, considering the
     * eligibility trace for state-action pairs and the specified transition. It refines the policy
     * using the updated Q-values.
     *
     * @param Q the Q-function representing the expected cumulative rewards for state-action pairs.
     * @param transition the transition information containing the current state, action, reward,
     * next state, and a flag indicating if the episode has ended.
     * @return the updated Q-function after applying the TD-Lambda update and improving the policy.
     */
    override fun invoke(
        Q: QFunction<State, Action>,
        transition: Transition<State, Action>
    ): QFunction<State, Action> {
        val (s, a, _, sPrime) = transition
        val done = transition.done
        val aPrime = if (!done) policy(sPrime) else null

        val delta = td(Q, transition, aPrime ?: a, gamma, done)
        trace = trace.decay(gamma, lambda()).update(s, a)
        var updatedQ = Q
        trace.values().forEach { (key, traceValue) ->
            val (state, action) = key
            val newQ = updatedQ[state, action] + alpha() * delta * traceValue
            updatedQ = updatedQ.update(state, action, newQ)
        }

        if (done) trace = trace.clear()
        policy = policy.improve(updatedQ)
        return updatedQ
    }
}