package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Implements the Bellman Q-function iteration algorithm for solving Markov Decision Processes (MDPs).
 *
 * This class performs dynamic programming to estimate the optimal Q-function for a given environment
 * and outputs a derived policy. The iterative process uses the Bellman equation to update state-action
 * value estimates until convergence, as defined by a specified convergence threshold (`theta`).
 *
 * @param State the type representing states in the Markov Decision Process.
 * @param Action the type representing actions that can be performed in the states.
 * @constructor Creates an instance of the BellmanQFunctionIteration algorithm.
 *
 * @param initialQ the initial Q-function from which values will be iteratively updated.
 * @param model the Markov Decision Process model describing the environment's dynamics.
 * @param gamma the discount factor used to weigh future rewards, where `0 <= gamma <= 1`.
 * @param theta the convergence threshold; iteration stops when the maximum update value is below this threshold.
 * @param stateActions a function that retrieves all possible actions for a given state.
 * @param onQFunctionUpdate an optional callback invoked whenever the Q-function is updated.
 */
class BellmanQFunctionIteration<State, Action>(
    private val initialQ: EnumerableQFunction<State, Action>,
    private val model: MDPModel<State, Action>,
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActions: StateActions<State, Action>,
    private val onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
) : DPIteration<State, Action>() {

    /**
     * Implements the Bellman Optimality Iteration algorithm for estimating the optimal
     * policy in a Markov Decision Process (MDP). The method iteratively updates the
     * Q-function based on the Bellman optimality equation until convergence is achieved
     * or the specified threshold (theta) is satisfied.
     *
     * @return A policy derived from the final Q-function, which selects the optimal
     * action for each state based on the maximum Q-value. If multiple actions have
     * the same Q-value, one is chosen randomly.
     */
    override fun plan(): Policy<State, Action> {
        var delta: Double
        var Q = initialQ

        do {
            delta = 0.0
            var newQ = Q

            for (s in model.allStates()) {
                val actions = stateActions(s)
                for (a in actions) {
                    val transitions = model.transitions(s, a)
                    val expectedValue = transitions.sumOf { t ->
                        val maxQNext = if (t.done) 0.0 else {
                            val nextActions = stateActions(t.nextState)
                            nextActions.maxOfOrNull { Q[t.nextState, it] } ?: 0.0
                        }
                        t.probability * (t.reward + gamma * maxQNext)
                    }

                    delta = maxOf(delta, kotlin.math.abs(Q[s, a] - expectedValue))
                    newQ = newQ.update(s, a, expectedValue)
                }
            }

            Q = newQ
            onQFunctionUpdate(Q)
        } while (delta > theta)

        return Policy { s ->
            val actions = stateActions(s)
            actions.maxByOrNull { a -> Q[s, a] } ?: actions.random()
        }
    }
}
