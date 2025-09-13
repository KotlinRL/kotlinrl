package io.github.kotlinrl.tabular

import io.github.kotlinrl.tabular.dp.*

/**
 * Applies the Policy Iteration algorithm to compute the optimal policy for a given environment.
 *
 * The Policy Iteration algorithm alternates between policy evaluation and policy improvement steps
 * to find the optimal policy and corresponding value function for a Markov Decision Process (MDP).
 * Policy evaluation computes the value function for the current policy, while policy improvement
 * updates the policy based on the computed value function. The process is repeated until convergence.
 *
 * @param theta The convergence threshold for the policy iteration process. The algorithm
 *              stops when the difference between consecutive policies is less than this value.
 *              Defaults to 1e-6.
 * @return A PolicyPlanner instance configured to perform policy iteration with the given
 *         convergence threshold.
 */
fun policyIteration(
    theta: Double = 1e-6
): PolicyPlanner = PolicyIteration(theta)

/**
 * Creates an instance of the Value Iteration algorithm with a specified convergence threshold.
 *
 * Value Iteration is a dynamic programming algorithm used to compute an optimal policy
 * and value function for a given Markov Decision Process (MDP). The algorithm iteratively updates
 * the value function for states until the change in values is less than the specified threshold (theta).
 *
 * @param theta the convergence threshold for value updates. The algorithm stops iterating
 *              when the maximum difference between successive value updates is less than this value.
 *              Default value is 1e-6.
 * @return a PolicyPlanner implementation based on the Value Iteration algorithm to solve MDPs.
 */
fun valueIteration(
    theta: Double = 1e-6
): PolicyPlanner = ValueIteration(theta)