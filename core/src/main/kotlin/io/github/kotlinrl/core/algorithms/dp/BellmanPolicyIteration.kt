package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

/**
 * Implements the Bellman Policy Iteration algorithm for solving
 * discrete Markov Decision Processes (MDPs). This algorithm alternates
 * between policy evaluation using the Bellman expectation equation
 * and policy improvement to converge to an optimal policy.
 *
 * @param State the type representing states in the MDP.
 * @param Action the type representing actions in the MDP.
 * @param initialPolicy the starting policy used for the iteration.
 * @param initialV the initial value function estimate.
 * @param model the MDP model used for state transitions and rewards.
 * @param gamma the discount factor in the range [0, 1), which determines the importance of future rewards.
 * @param theta the convergence threshold for the value function updates.
 * @param stateActions a function that determines the available actions for a given state.
 * @param onValueFunctionUpdate a callback that executes whenever the value function is updated.
 * @param onPolicyUpdate a callback that triggers whenever the policy is updated.
 */
class BellmanPolicyIteration<State, Action>(
    private var initialPolicy: Policy<State, Action>,
    private var initialV: EnumerableValueFunction<State>,
    private val model: MDPModel<State, Action>,
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActions: StateActions<State, Action>,
    private val onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
    private val onPolicyUpdate: PolicyUpdate<State, Action> = { }
) : DPIteration<State, Action>() {

    /**
     * A `BellmanValueFunctionEstimator` instance used for estimating the value function
     * by applying the Bellman update equation.
     *
     * The `estimator` is responsible for iteratively updating the value function
     * of the states based on expected rewards and transitions, guided by the discount factor `gamma`.
     *
     * This variable is utilized within the `BellmanPolicyIteration` class to perform
     * dynamic programming-based policy iteration.
     *
     * @see BellmanValueFunctionEstimator
     */
    private val estimator = BellmanValueFunctionEstimator<State, Action>(gamma)

    /**
     * Performs the Bellman Policy Iteration algorithm to find the optimal policy for
     * a given Markov Decision Process (MDP). The method iteratively evaluates and improves
     * the current policy until convergence is achieved.
     *
     * The algorithm alternates between two main steps:
     * 1. Policy Evaluation: Evaluates the current policy using the value function update rule.
     * 2. Policy Improvement: Generates a new policy by selecting actions that maximize expected
     *    rewards based on the current value function.
     *
     * Convergence is achieved when the policy is stable, meaning no further changes occur
     * during the policy improvement step.
     *
     * @return The optimal policy derived through the Bellman Policy Iteration process.
     */
    override fun plan(): Policy<State, Action> {
        var policy = initialPolicy
        var V = initialV
        var stable: Boolean

        do {
            var delta: Double
            do {
                delta = 0.0
                val transitions = model.allStates().flatMap { s ->
                    model.transitions(s, policy(s))
                }

                val updatedV = estimator.estimate(V, transitions)

                for (s in updatedV.allStates()) {
                    delta = maxOf(delta, abs(updatedV[s] - V[s]))
                }

                V = updatedV
                onValueFunctionUpdate(V)
            } while (delta > theta)

            stable = true
            val newPolicy = Policy<State, Action> { s ->
                val actions = stateActions(s)
                val best = actions.maxByOrNull { a ->
                    model.transitions(s, a).sumOf { t ->
                        t.probability * (t.reward + gamma * if (t.done) 0.0 else V[t.nextState])
                    }
                }
                best ?: actions.random()
            }

            for (s in model.allStates()) {
                if (policy(s) != newPolicy(s)) {
                    stable = false
                    break
                }
            }

            policy = newPolicy
            onPolicyUpdate(policy)

        } while (!stable)

        return policy
    }
}

