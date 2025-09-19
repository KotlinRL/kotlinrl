package io.github.kotlinrl.tabular

import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.core.distributions.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*
import kotlin.random.*

/**
 * Type alias representing the value function table (V-Table).
 *
 * A V-Table is a one-dimensional array that stores the value of each state
 * in a Markov Decision Process (MDP). Each entry in the table corresponds
 * to the expected cumulative reward starting from a particular state,
 * following a specific policy.
 *
 * This alias is used to simplify and standardize the representation of
 * value function tables in tabular reinforcement learning algorithms.
 */
typealias VTable = D1Array<Double>
/**
 * A type alias representing a callback function for receiving the V-Table when it is updated
 *
 * @param VTable Represents the updated state-value table to be handled by the callback.
 */
typealias VTableUpdate = (VTable) -> Unit
/**
 * Type alias representing the policy table (P-Table).
 *
 * A P-Table is a one-dimensional array where each entry corresponds to the action
 * chosen for a given state in a Markov Decision Process (MDP). It essentially
 * represents a deterministic policy as a direct mapping from states to actions.
 *
 * This alias is used to simplify and standardize the representation of policies
 * in tabular reinforcement learning algorithms or decision-making processes.
 */
typealias PTable = D1Array<Int>
/**
 * A type alias representing a callback function for receiving the P-Table (policy table) when it is updated.
 *
 * @param PTable Represents the updated policy table to be handled by the callback.
 */
typealias PTableUpdate = (PTable) -> Unit
/**
 * A type alias representing a 2D array of `Double` values used as a Q-table
 * in reinforcement learning algorithms.
 *
 * The `QTable` is commonly employed to store and update the estimated action-value
 * function (Q-values) for state-action pairs. Each row corresponds to a specific state,
 * and each column corresponds to a specific action, with entries representing the
 * estimated value of executing an action in a given state.
 *
 * This alias simplifies and standardizes references to Q-tables in the library and
 * ensures compatibility with operations requiring 2D arrays of type `Double`.
 */
typealias QTable = D2Array<Double>
/**
 * A type alias representing a callback function for receiving the Q-table when it is updated.
 **/
typealias QTableUpdate = (QTable) -> Unit

/**
 * Constructs an epsilon-greedy policy for selecting actions based on the provided Q-table.
 * Epsilon-greedy is a strategy combining exploration and exploitation in reinforcement learning.
 * With probability epsilon, the policy selects a random action (exploration),
 * and with probability (1 - epsilon), it selects the action with the highest Q-value (exploitation).
 *
 * @param epsilon A parameter schedule controlling the exploration rate (epsilon). Dynamic adjustments
 *                to epsilon allow for balancing exploration and exploitation over time, often starting
 *                with higher exploration and transitioning to exploitation as the learning process continues.
 * @param rng A random number generator used for sampling during exploration. Ensures stochasticity
 *            in the selection of random actions.
 * @return A policy implementing the epsilon-greedy approach. This policy maps states to actions,
 *         either deterministically choosing the action with the highest Q-value or probabilistically
 *         exploring based on the epsilon parameter.
 */
fun QTable.epsilonGreedy(
    epsilon: ParameterSchedule,
    rng: Random = Random.Default,
    tieTol: Double = 1e-12,
): Policy<Int, Int> {
    val Q = this;

    fun argmaxTies(row: MultiArray<Double, D1>): List<Int> {
        val m = row.max()!!
        val ties = row.indices.filter { abs(row[it] - m) <= tieTol }
        return ties
    }

    return object : Policy<Int, Int> {
        override fun invoke(state: Int): Int {
            return this[state].sample(rng)
        }

        override fun get(state: Int): Distribution<Int> {
            val row = Q[state]
            val n = row.size
            require(n > 0) { "No actions available" }
            val ties = argmaxTies(row)
            val k = ties.size
            val (epsilon) = epsilon()
            val e = epsilon.coerceIn(0.0, 1.0)
            if (e == 0.0 || n == 1) {
                val pick = if (k == 1) ties[0] else ties[rng.nextInt(k)]
                return Distribution.delta(pick)
            } else {
                val base = e / n
                val probs = mk.d1array<Double>(n) { base }
                val extra = (1.0 - e) / k
                for (i in ties) probs[i] = base + extra
                return Distribution.categorical((0 until n).toList(), probs)
            }
        }
    }
}

/**
 * Creates a greedy policy based on the Q-table.
 *
 * The greedy policy selects the action with the highest Q-value for a given state.
 * This is done deterministically, where the action corresponding to the maximum
 * Q-value from the provided Q-table is chosen.
 *
 * @return a Policy that maps states to actions by selecting the action
 * with the highest Q-value for each state.
 */
fun QTable.greedy(): Policy<Int, Int> {
    val Q = this

    return object : Policy<Int, Int> {
        override fun invoke(state: Int): Int {
            return Q[state].argMax()
        }

        override fun get(state: Int): Distribution<Int> {
            return Distribution.delta(this(state))
        }
    }
}

/**
 * Converts a `PTable` representation of a policy into a `Policy` object.
 *
 * The resulting `Policy` object provides a deterministic action mapping for
 * any given state based on the provided `PTable`. Additionally, it offers
 * the ability to fetch a delta distribution, where all probability mass is concentrated
 * on the action chosen for the given state.
 *
 * @return A `Policy` instance that encapsulates the state-to-action mapping defined
 *         by the `PTable`.
 */
fun PTable.pi(): Policy<Int, Int> {
    val pi = this

    return object : Policy<Int, Int> {
        override fun invoke(state: Int): Int {
            return pi[state]
        }

        override fun get(state: Int): Distribution<Int> {
            return Distribution.delta(this(state))
        }
    }
}

