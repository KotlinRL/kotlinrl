package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

/**
 * Implements the Off-Policy Monte Carlo estimation of the state-value function (V)
 * using importance sampling and trajectory data.
 *
 * This algorithm updates an enumerable value function (V) to reflect the estimates derived
 * from a given trajectory while adjusting for discrepancies between the behavior policy (which generates
 * the trajectory) and the target policy (for which estimates are being calculated).
 *
 * The key mechanism used to reconcile the differences between the behavior and target policies
 * is importance sampling. This adjusts the contributions of trajectory data based on the likelihood
 * of the observed actions under the behavior policy compared to the target policy.
 *
 * Key characteristics include:
 * - Works with a stochastic behavior policy and an arbitrary target policy.
 * - Continuously updates and stores cumulative weights for states to normalize value estimates.
 * - Accounts for the discounted return (G) and uses an iterative backward traversal of the trajectory.
 * - Halts updates if the importance sampling weight becomes zero or when a mismatch occurs between
 *   the actions in the trajectory and the target policy.
 *
 * @param State the type representing the environment's states.
 * @param Action the type representing the actions taken in the environment.
 * @param gamma the discount factor for computing the cumulative discounted return G. Represents the influence
 *        of future rewards, with values in the range [0, 1].
 * @param behaviorPolicy the stochastic policy used to generate the trajectory. This is compared against the
 *        target policy to compute importance sampling ratios.
 * @param targetPolicy the deterministic or stochastic policy being evaluated and improved. Updated estimates
 *        are computed for this policy's behavior.
 */
class OffPolicyMonteCarloEstimateV_fromTrajectory<State, Action>(
    private val gamma: Double,
    private val behaviorPolicy: StochasticPolicy<State, Action>,
    private var targetPolicy: Policy<State, Action>,
) : EstimateV_fromTrajectory<State, Action> {

    private val C: MutableMap<Comparable<*>, Double> = mutableMapOf()

    /**
     * Invokes the off-policy Monte Carlo estimation method for updating a value function based on
     * a given trajectory and importance sampling. This method iterates through the trajectory in reverse,
     * applying corrections based on the behavior policy and target policy to estimate the updated value function.
     *
     * @param V the initial value function that estimates the scalar value for each state.
     * @param trajectory the trajectory to be used for updating the value function. It consists of states,
     * actions, and rewards collected during an episode.
     * @return the updated value function after processing the given trajectory using importance sampling adjustments.
     */
    override fun invoke(V: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State> {
        var G = 0.0
        var W = 1.0
        var newV = V

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = s.toComparable()

            val c = C.getOrDefault(key, 0.0) + W
            C[key] = c

            val oldV = newV[s]
            val newValue = oldV + (W / c) * (G - oldV)
            newV = newV.update(s, newValue)

            if (a != targetPolicy(s)) break

            val prob = behaviorPolicy.probability(s, a)
            if (prob == 0.0) break
            W /= prob
        }

        return newV
    }
}
