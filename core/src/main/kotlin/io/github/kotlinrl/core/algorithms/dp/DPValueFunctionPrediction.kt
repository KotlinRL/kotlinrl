package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Class for performing Dynamic Programming (DP) value function prediction using
 * a specified Markov Decision Process (MDP) model and an estimator.
 *
 * The `DPValueFunctionPrediction` class predicts the state-value function (`V`)
 * for a given policy by iteratively refining the function with the help of an
 * `DPValueFunctionEstimator`. This approach relies on DP techniques to compute
 * and update the value function for all states in the MDP. The prediction involves:
 * - Iterating through all states and generating their corresponding transitions using the model.
 * - Using the estimator to update the state-value function based on the provided policy's
 *   state-action mappings and the trajectory from the MDP model.
 *
 * @param State the type representing states in the environment or MDP.
 * @param Action the type representing actions available in the environment or MDP.
 * @constructor Initializes a new instance of `DPValueFunctionPrediction` with a given
 * `EnumerableValueFunction`, MDP model, and value function estimator.
 *
 * @property V the current state-value function being predicted, starting
 * with the provided initial function and refined over iterations.
 */
class DPValueFunctionPrediction<State, Action>(
    initialV: EnumerableValueFunction<State>,
    private val model: MDPModel<State, Action>,
    private val estimator: DPValueFunctionEstimator<State, Action>
) {

    /**
     * Represents the current state-value function (`V`) being predicted and refined
     * within the dynamic programming process for a Markov Decision Process (MDP).
     *
     * This variable maps states in the MDP to their respective value estimates, reflecting
     * the expected return starting from each state and following the given policy. It is
     * iteratively updated using a provided dynamic programming estimator and the transitions
     * generated from the MDP model.
     *
     * Initially, `V` is set to the provided `initialV`, serving as the starting point for
     * the value function prediction. Subsequent updates to `V` during each prediction iteration
     * refine its accuracy, aiming to better approximate the true state-value function under
     * the specified policy.
     *
     * This property is read-only from outside the class and can only be modified internally
     * during the value function prediction process.
     */
    var V: EnumerableValueFunction<State> = initialV
        private set

    /**
     * Performs dynamic programming to estimate and refine the state-value function (V-function)
     * based on the provided policy and the Markov Decision Process (MDP) model.
     * The method iterates through all states, computes state-action transitions using the policy,
     * and updates the value function using the estimator.
     *
     * @param policy The policy defining the action to take for each state. It maps states to actions.
     * @return The updated state-value function, representing the estimated values for each state
     *         after applying the dynamic programming update.
     */
    operator fun invoke(policy: Policy<State, Action>): EnumerableValueFunction<State> {
        val transitions = model.allStates().flatMap { s ->
            model.transitions(s, policy(s))
        }

        V = estimator.estimate(V, transitions)
        return V
    }
}
