package io.github.kotlinrl.core.algorithms.td.lambda

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*

/**
 * Implementation of a Q-function estimator that utilizes the TD(λ) algorithm with eligibility traces
 * to update Q-values based on observed state transitions in reinforcement learning environments.
 *
 * The TD(λ) algorithm combines the temporal difference learning and eligibility traces to balance
 * between one-step TD updates and Monte Carlo updates. This provides a mechanism to consider both
 * immediate and extended past experiences in the learning process.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @property initialPolicy the policy used to choose actions in future states during learning.
 * @property alpha the schedule determining the learning rate for Q-value updates.
 * @property lambda the schedule that defines the trace decay rate for eligibility traces.
 * @property gamma the discount factor that controls the importance of future rewards.
 * @property td the temporal difference error function used for calculating the TD-error.
 * @property initialEligibilityTrace the initial state of the eligibility trace.
 * @property onEligibilityTraceUpdate a callback function invoked whenever the eligibility trace is updated.
 */
class TDLambdaQFunctionEstimator<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val alpha: ParameterSchedule,
    private val lambda: ParameterSchedule,
    private val gamma: Double,
    private val td: TDQError<State, Action>,
    initialEligibilityTrace: EligibilityTrace<State, Action>,
    private val onEligibilityTraceUpdate: EligibilityTraceUpdate<State, Action> = { }
) : TransitionQFunctionEstimator<State, Action> {

    /**
     * Represents the policy used for selecting actions in the reinforcement learning algorithm.
     * The policy defines a mapping from states to actions and serves as a framework for decision-making
     * throughout the learning process. This variable holds the current policy and is updated dynamically
     * as the Q-function improves during training.
     *
     * The initial value is `initialPolicy`, which provides a starting point for action selection.
     * Updates to this variable reflect changes made during policy improvement steps, based on the
     * temporal-difference learning method being applied.
     */
    var policy: Policy<State, Action> = initialPolicy
    /**
     * Tracks the eligibility trace state and manages updates to the underlying eligibility trace structure.
     *
     * The `trace` variable represents the current eligibility trace used during computation in temporal-difference
     * learning methods. Eligibility traces serve as a mechanism for assigning credit to state-action pairs based
     * on their temporal proximity and relevance to observed transitions.
     *
     * Upon being set, the trace value triggers the `onEligibilityTraceUpdate` callback, enabling external
     * components to respond to or monitor changes in the eligibility trace. This is particularly useful for debugging,
     * visualization, or any custom operations involving the trace dynamics.
     *
     * This variable is initialized to the `initialEligibilityTrace`, which reflects the state-action relevance
     * structure specified during the creation of the estimator. The setter ensures changes are actively managed
     * with respect to the registered callback.
     *
     * @property trace The current eligibility trace used to manage state-action pair credit assignment dynamics.
     *                Initial value is specified through `initialEligibilityTrace`.
     */
    private var trace: EligibilityTrace<State, Action> = initialEligibilityTrace
        set(value) {
            field = value
            onEligibilityTraceUpdate(value)
        }

    /**
     * Estimates and updates the Q-values based on the provided transition and eligibility traces,
     * applying the TD(λ) update rule.
     *
     * @param Q the current Q-function representing state-action value estimates.
     * @param transition the transition information consisting of the current state, action, reward,
     * and resulting state, as well as whether the episode has terminated.
     * @return the updated Q-function after applying the TD(λ) update rule.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
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

        return updatedQ
    }
}