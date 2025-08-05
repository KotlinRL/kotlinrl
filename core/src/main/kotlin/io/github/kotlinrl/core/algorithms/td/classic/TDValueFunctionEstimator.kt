package io.github.kotlinrl.core.algorithms.td.classic

import io.github.kotlinrl.core.*

/**
 * Implements a Temporal Difference (TD) value function estimator for reinforcement learning.
 *
 * This class is responsible for computing the updated value function based on observed transitions
 * using a specified TD error calculation method, learning rate, and discount factor. The update is
 * applied iteratively, allowing dynamic refinement of the value function estimates over time.
 *
 * The estimator utilizes TD error computation to refine the existing value function by incorporating
 * the observed reward and expected future rewards from the given transition. It is commonly employed
 * in reinforcement learning algorithms where value function updates are required.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param alpha a schedule defining the learning rate, which adjusts the step size of updates over time.
 * @param gamma the discount factor for future rewards, ranging between 0 and 1.
 * @param td a TD error calculation method used to compute the temporal difference.
 * Defaults to an on-policy TD(0) approach.
 */
class TDValueFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: TDVError<State> = TDVErrors.tdZero()
) : TransitionValueFunctionEstimator<State, Action> {

    /**
     * Estimates an updated enumerable value function based on the provided transition.
     *
     * This method applies a Temporal Difference (TD) update to a given enumerable value function
     * using the observed transition and the specified learning rate and discount factor. The update
     * is computed only if the TD error (delta) is non-zero.
     *
     * @param V the enumerable value function representing the estimated scalar values for states.
     * @param transition the transition containing the current state, action, and observed reward.
     * @return the updated enumerable value function reflecting the applied TD update.
     */
    override fun estimate(
        V: EnumerableValueFunction<State>,
        transition: Transition<State, Action>
    ): EnumerableValueFunction<State> {
        val (s, _, _) = transition
        val delta = td(V, transition, gamma)
        if (delta == 0.0) return V
        return V.update(s, V[s] + alpha() * delta)
    }
}
