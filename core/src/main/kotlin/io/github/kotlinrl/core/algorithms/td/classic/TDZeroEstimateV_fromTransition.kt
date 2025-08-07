package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements the TD(0) algorithm for value function updates after observing a state transition in reinforcement learning.
 *
 * TD(0) is a Temporal Difference learning method that updates the value function incrementally using observed transitions
 * and applies a one-step lookahead. This class combines a defined learning rate schedule (`alpha`) and discount factor
 * (`gamma`) with a TD error computation strategy (`td`) to estimate the updated value function.
 *
 * The update mechanism works as follows:
 * 1. The current TD error is computed using the provided transition, value function, and discount factor.
 * 2. If the TD error is non-zero, the value of the current state is updated using the learning rate schedule and the TD error.
 * 3. The updated value function is returned.
 *
 * @param State the type representing the state space of the environment.
 * @param Action the type representing the action space of the environment.
 * @param alpha the learning rate schedule, determining the step size for updates.
 * @param gamma the discount factor, controlling the weight of future rewards (must be in the range [0, 1]).
 * @param td the strategy for calculating the TD error. Defaults to `tdZero`, which represents the classical TD(0) method.
 */
class TDZeroEstimateV_fromTransition<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDVError<State> = TDVErrors.tdZero()
) : EstimateV_fromTransition<State, Action> {

    /**
     * Updates the given value function based on the observed state transition using the TD(0) learning method.
     *
     * This function applies the Temporal Difference (TD) algorithm to compute the TD error
     * and update the value of the current state if the computed error is non-zero. The updated
     * value is determined by incrementally adjusting the current value based on the TD error
     * and the learning rate schedule.
     *
     * @param V the value function to be updated, which maps states to estimated scalar values.
     * @param transition the observed transition containing the current state, reward, next state,
     * and a terminal flag indicating if the episode has ended.
     * @return the updated value function after applying the TD(0) learning rule.
     */
    override fun invoke(
        V: ValueFunction<State>,
        transition: Transition<State, Action>
    ): ValueFunction<State> {
        val (s, _, _) = transition
        val delta = td(V, transition, gamma)
        if (delta == 0.0) return V
        return V.update(s, V[s] + alpha() * delta)
    }
}
