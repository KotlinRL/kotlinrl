package io.github.kotlinrl.tabular.dp

import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.core.model.TabularMDP
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

/**
 * Implements the Value Iteration algorithm for solving Markov Decision Processes (MDPs).
 *
 * Value Iteration is a dynamic programming technique used to compute an optimal policy
 * and corresponding value function for an MDP. The algorithm iteratively updates the
 * state value function by applying the Bellman optimality equation until convergence.
 *
 * This implementation assumes a finite state and action space and leverages the
 * Bellman backup operator for value updates.
 *
 * @param theta a small threshold value representing the stopping criterion for
 *              value updates. The iterations stop when the largest difference
 *              between successive value function updates is less than this threshold.
 */
class ValueIteration(
    private val theta: Double = 1e-6,
) : PolicyPlanner {

    /**
     * Solves a Markov Decision Process (MDP) using the Value Iteration algorithm.
     *
     * The method iteratively updates the value function for each state until the largest change
     * in value updates falls below a pre-defined threshold (theta). Once convergence is achieved,
     * it generates a pi policy based on the final Q-function.
     *
     * @param MDP The Markov Decision Process (MDP) model to be solved. This includes the states,
     *            actions, transition probabilities, reward function, and discount factor.
     * @return A pair consisting of:
     *         - A pi policy derived from the final Q-function.
     *         - The final value function as a mapping from states to their values.
     */
    override fun invoke(MDP: TabularMDP): Pair<PTable, VTable> {
        val (_, _, R, T, gamma)  = MDP
        val S = T.shape[0]
        val A = T.shape[1]
        var V = mk.zeros<Double>(A)
        val Q = mk.zeros<Double>(S, A)
        var norm = Double.POSITIVE_INFINITY
        while (norm > theta) {
            val newV = mk.ndarray(IntRange(0, S).map { s ->
                IntRange(0, A).maxOf { a ->
                    val backup = bellmanBackup(s, a, R, T, gamma, V)
                    Q[s, a] = backup
                    backup
                }
            })
            val diff = V - newV
            norm = IntRange(0, diff.size).maxOf { abs(diff[it]) }
            V = newV
        }
        val policy = mk.ndarray(IntRange(0, S).map { Q[it].argMax() })
        return policy to V
    }
}