package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

/**
 * BellmanValueFunctionIteration is an implementation of the value iteration algorithm for solving
 * Markov Decision Processes (MDPs). This algorithm iteratively improves the estimated value function
 * using the Bellman update equation until the changes between iterations are below a specified threshold.
 *
 * @param State the type representing states in the Markov Decision Process.
 * @param Action the type representing actions that can be performed in the Markov Decision Process.
 * @param initialV the initial value function representing an estimate of state values.
 * @param model the Markov Decision Process (MDP) model defining the environment dynamics and reward structure.
 * @param gamma the discount factor used to weigh future rewards, ranging between 0 (immediate rewards) and 1 (long-term rewards).
 * @param theta the threshold value for convergence, representing the minimum change in value function updates required to stop iterations.
 * @param stateActions a function providing the list of available actions for a specific state.
 * @param onValueFunctionUpdate a callback function invoked after each value function update, allowing inspection or side-effects during the process.
 */
class BellmanValueFunctionIteration<State, Action>(
    private var initialV: EnumerableValueFunction<State>,
    private var model: MDPModel<State, Action>,
    private var gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActions: StateActions<State, Action>,
    private val onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
) : DPIteration<State, Action>() {

    /**
     * Executes the Bellman Value Function Iteration algorithm to compute the optimal policy
     * for a given Markov Decision Process (MDP) model. The algorithm iteratively updates the
     * state-value function until the estimated values converge within a defined threshold.
     *
     * The optimal policy is derived based on the final value function, selecting*/
    override fun plan(): Policy<State, Action> {
        var delta: Double
        var V = initialV

        do {
            delta = 0.0
            var newV = V

            for (s in model.allStates()) {
                val actions = stateActions(s)
                if (actions.isEmpty()) continue

                val bestActionValue = actions.maxOf { a -> expectedReturn(s, a, V) }
                delta = maxOf(delta, abs(bestActionValue - V[s]))
                newV = newV.update(s, bestActionValue)
            }

            V = newV
            onValueFunctionUpdate(V)

        } while (delta > theta)

        return Policy { s ->
            val actions = stateActions(s)
            actions.maxByOrNull { a -> expectedReturn(s, a, V) } ?: actions.random()
        }
    }

    private fun expectedReturn(s: State, a: Action, V: EnumerableValueFunction<State>): Double {
        return model.transitions(s, a).sumOf { t ->
            t.probability * (t.reward + gamma * if (t.done) 0.0 else V[t.nextState])
        }
    }
}
