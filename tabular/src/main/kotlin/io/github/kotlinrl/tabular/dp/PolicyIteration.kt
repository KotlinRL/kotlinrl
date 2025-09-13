package io.github.kotlinrl.tabular.dp

import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.core.model.TabularMDP
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

/**
 * Implements the Policy Iteration algorithm for solving a given Markov Decision Process (MDP).
 *
 * Policy Iteration is an iterative method used to compute the optimal policy and value function
 * for a given environment. It alternates between two key steps:
 * - Policy Evaluation: Computes the value function for a fixed policy.
 * - Policy Improvement: Updates the policy based on the computed value function.
 *
 * The algorithm continues this alternating process until convergence, where no further updates
 * to the policy are needed.
 *
 * @property theta A convergence threshold. The algorithm iterates until the difference between
 *                 consecutive policies is less than this value.
 */
class PolicyIteration(
    private val theta: Double,
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
    override fun invoke(MDP: TabularMDP): Pair<PTable, VTable> {
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
        return policy to V
    }
}