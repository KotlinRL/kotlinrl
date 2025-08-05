package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents an implementation of a transition observer that maintains and updates an enumerable
 * value function based on observed state-action transitions. This implementation leverages a
 * `TransitionValueFunctionEstimator` to compute updates to the value function dynamically.
 *
 * This class is designed for use in reinforcement learning scenarios where value function
 * estimation is essential for evaluating the expected utility of states within the environment.
 * The value function (`V`) is updated incrementally as transitions are observed.
 *
 * @param State the type representing the state space of the environment.
 * @param Action the type representing the action space of the environment.
 * @param initialV the initial enumerable value function to be maintained and updated.
 * @param estimator the transition value function estimator that computes updates
 * to the value function based on observed transitions.
 */
class TransitionValueFunctionPrediction<State, Action>(
    initialV: EnumerableValueFunction<State>,
    private val estimator: TransitionValueFunctionEstimator<State, Action>,
) : TransitionObserver<State, Action> {

    /**
     * The enumerable value function (`V`) representing the current state-value mappings in the context
     * of a reinforcement learning environment. This property is initialized with an `initialV` value
     * function and is updated dynamically via the `TransitionValueFunctionEstimator` as transitions
     * are observed. Updates are performed based on the estimated value of observed state-action transitions.
     *
     * The `V` property is read-only from outside the `TransitionValueFunctionPrediction` class and is
     * only modified internally to ensure controlled updates to the value function.
     */
    var V = initialV
        private set

    /**
     * Invokes the transition observer to process the given state-action transition and update
     * the maintained enumerable value function using the specified estimator. The value function
     * is updated based on the result of the estimation process, which incorporates the effects
     * of the observed transition on the expected state utility.
     *
     * @param transition the observed transition containing the initial state, the action taken,
     * the resulting state, and the associated reward. This transition is used to estimate
     * updates to the enumerable value function.
     */
    override operator fun invoke(transition: Transition<State, Action>) {
        V = estimator.estimate(V, transition)
    }
}