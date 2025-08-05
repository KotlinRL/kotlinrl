package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionEstimator

/**
 * An implementation of off-policy Monte Carlo Q-function estimation for reinforcement learning.
 *
 * This class estimates Q-values based on trajectories collected under a behavior policy while
 * updating the Q-values to conform with a target policy. Importance sampling is used to ensure
 * corrections for the discrepancy between the behavior policy and target policy.
 *
 * The underlying algorithm follows the standard off-policy Monte Carlo update rules:
 * - Backtrack through each trajectory, computing the return `G` for each encountered state-action pair.
 * - Use cumulative weights `W` derived from the importance sampling ratio to adjust updates.
 * - Update the Q-function iteratively using weighted returns and a visitation count-based approach.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initTargetPolicy the initial target policy which the estimated Q-function aims to conform with.
 * @param behaviorPolicy the policy under which trajectories are generated (used for importance sampling).
 * @param gamma the discount factor (0 ≤ gamma ≤ 1) used to weight future rewards.
 */
class OffPolicyMonteCarloQFunctionEstimator<State, Action>(
    initTargetPolicy: QFunctionPolicy<State, Action>,
    private val behaviorPolicy: QFunctionPolicy<State, Action>,
    private val gamma: Double,
) : TrajectoryQFunctionEstimator<State, Action> {
    private val C: MutableMap<StateActionKey<*, *>, Double> = mutableMapOf()

    /**
     * The policy being optimized and targeted by the off-policy Monte Carlo control algorithm.
     *
     * `targetPolicy` represents the ideal policy that the algorithm aims to improve and converge towards,
     * utilizing trajectories sampled from a separate behavioral policy. It is updated based on the estimated
     * Q-function values and guides the agent's decision-making towards optimal behavior in the environment.
     *
     * This policy is iteratively improved by combining the Q-function estimates and applying a policy improvement
     * strategy. The updates ensure that the algorithm leverages importance sampling to correct for the mismatch
     * between the behavioral and target policies while maintaining convergence towards the optimal policy.
     */
    var targetPolicy: Policy<State, Action> = initTargetPolicy

    /**
     * Estimates the updated Q-function using off-policy Monte Carlo control with importance sampling.
     *
     * This method processes a given trajectory of state-action-reward tuples, incrementally computes
     * the importance-sampled return, and updates the Q-function estimate for state-action pairs
     * in accordance with their cumulative rewards and weights. The updates consider the difference
     * between the cumulative return (G) and the previous Q-value, adjusting it based on importance sampling weights.
     *
     * @param Q the current Q-function that maps state-action pairs to Q-values.
     * @param trajectory the list of states, actions, and rewards representing an episode trajectory.
     * @return an updated instance of the Q-function with recalculated Q-values based on the trajectory.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        var G = 0.0
        var W = 1.0
        var currentQ = Q

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G

            val key = stateActionKey(s, a)
            val oldC = C.getOrDefault(key, 0.0)
            val newC = oldC + W
            C[key] = newC

            val oldQ = currentQ[s, a]
            val updatedQ = oldQ + (W / newC) * (G - oldQ)
            currentQ = currentQ.update(s, a, updatedQ)

            if (a != targetPolicy(s)) break
            val prob = behaviorPolicy.probability(s, a)
            if (prob == 0.0) break
            W /= prob
        }

        return currentQ
    }
}
