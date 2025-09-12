package io.github.kotlinrl.tabular.dp

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.*

/**
 * This object provides an implementation of the policy evaluation algorithm, a key component
 * in dynamic programming methods for solving Markov Decision Processes (MDPs).
 *
 * The policy evaluation algorithm estimates the value function for a given policy by
 * iteratively applying the Bellman backup equation. It continues this process until the
 * change in value estimates becomes smaller than a predefined threshold or a maximum number of
 * iterations is reached. This iterative approach enables the estimation of the expected
 * cumulative reward for states under the given policy.
 */
object policyEvaluation {

    /**
     * Invokes the value computation process based on a given policy, reward function,
     * transition probabilities, discount factor, and tolerance level.
     *
     * This function iteratively calculates the value function V(s) for all states
     * under a fixed policy until the difference between subsequent iterations (delta)
     * is less than the specified tolerance level (tol).
     *
     * @param policy The policy represented as a 1D array of integers, where each
     *               element maps a state to a corresponding action.
     * @param R A 2D array representing the reward function, where R[s, a] is the
     *          reward for taking action a in state s.
     * @param T A 3D array representing the transition probabilities, where T[s, a, s']
     *          is the probability of transitioning to state s' after taking action a
     *          in state s.
     * @param gamma The discount factor, a value between 0 and 1, which determines
     *              the importance of future rewards.
     * @param tol The tolerance level for convergence, by default set to 1e-3. The
     *            iteration stops when the maximum change in value function (delta)
     *            falls below this value.
     * @return A 1D array of doubles representing the computed value function V(s)
     *         for each state under the given policy.
     */
    operator fun invoke(
        policy: D1Array<Int>,
        R: D2Array<Double>,
        T: D3Array<Double>,
        gamma: Double,
        tol: Double = 1e-3
    ): D1Array<Double> {
        val S = T.shape[0]
        val A = T.shape[1]
        val V = mk.zeros<Double>(S)

        var delta = Double.POSITIVE_INFINITY
        while (delta > tol) {
            delta = 0.0
            val Q = mk.ndarray((0 until S).map { s ->
                (0 until A  ).map { a ->
                    bellmanBackup(s, a, R, T, gamma, V)
                }
            })
            for(s in 0 until S) {
                val oldVal = V[s]
                V[s] = Q[s, policy[s]]
                val newVal = V[s]
                delta = max(delta, abs(oldVal - newVal))
            }
        }
        return V
    }
}