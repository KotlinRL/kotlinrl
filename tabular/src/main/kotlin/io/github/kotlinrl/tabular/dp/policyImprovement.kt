package io.github.kotlinrl.tabular.dp

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*


/**
 * The `policyImprovement` object encapsulates the logic for deriving a pi policy
 * in the context of a Markov Decision Process (MDP). Using the provided value function,
 * it computes Q-values for state-action pairs and then forms a policy that selects
 * actions maximizing the Q-values for each state.
 */
object policyImprovement {

    operator fun invoke(R: D2Array<Double>, T: D3Array<Double>, V: D1Array<Double>, gamma: Double): D1Array<Int> {
        val S = T.shape[0]
        val A = T.shape[1]
        val Q = mk.ndarray(IntRange(0,S).map { s ->
            IntRange(0, A).map { a ->
                bellmanBackup(s, a, R, T, gamma, V)
            }
        })
        val policy = mk.ndarray((0 until S).map { s -> Q[s].argMax() })
        return policy
    }
}
