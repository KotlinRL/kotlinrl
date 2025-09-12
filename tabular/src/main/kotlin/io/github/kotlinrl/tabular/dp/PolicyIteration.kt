package io.github.kotlinrl.tabular.dp

import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.core.model.TabularMDP
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

/**
 * Implements the Policy Iteration algorithm for Markov Decision Processes (MDPs).
 *
 * Policy Iteration is an iterative method for generating optimal policies in
 * environments defined by MDP models. The algorithm alternates between two main
 * steps: policy evaluation and policy improvement. During policy evaluation,
 * the value function for the current policy is computed. In the policy improvement
 * step, the current policy is updated to be pi with respect to the evaluated
 * value function. This process continues until the policy converges to the optimal
 * policy, or a specified convergence threshold is met.
 *
 * Key features:
 * - Supports customizable convergence criteria through a threshold parameter (theta).
 * - Includes a configurable maximum number of iterations to ensure termination.
 * - Allows integration of a random number generator for generating random policies.
 *
 * @param State The type representing the states within the MDP.
 * @param Action The type representing the actions available within the MDP.
 * @param theta The convergence threshold for stopping policy iteration. The algorithm
 *              halts when the maximum change in the value function across all states
 *              is less than this threshold. Default is 1e-6.
 * @param maxIterations The maximum number of iterations allowed during policy evaluation
 *                      or improvement before termination. Default is 10000.
 * @param rng Random number generator used for creating initial random policies.
 */
class PolicyIteration(
    private val theta: Double = 1e-6,
) : PolicyPlanner {

    /**
     * Iteratively performs policy iteration for a given Markov Decision Process (MDP).
     * The process alternates between policy evaluation and policy improvement steps until
     * convergence, returning the optimal policy and value function for the MDP.
     *
     * @param MDP The Markov Decision Process model containing the state space, action space,
     *            reward function, transition probabilities, and discount factor.
     * @return A pair consisting of the optimal policy and the corresponding value function
     *         derived from the policy iteration algorithm.
     */
    override fun invoke(MDP: TabularMDP): Pair<Policy<Int, Int>, VTable> {
        val (_, _, R, T, gamma)  = MDP
        var V = mk.zeros<Double>(T.shape[0])
        var policy = mk.rand<Int, D1>(dims = intArrayOf(T.shape[0]), from = 0, until = T.shape[1])
        var i = 0
        var norm = 1.0
        while (i == 0 || norm > theta) {
            V = policyEvaluation(policy, R, T, gamma)
            val policyNew = policyImprovement(R, T, V, gamma)
            val diff = policy - policyNew
            norm = (0 until diff.size).sumOf { abs(diff[it]) }.toDouble()
            policy = policyNew
            i++
        }
        return policy.pi() to V
    }
}