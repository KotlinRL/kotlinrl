package io.github.kotlinrl.tabular.td.classic

import io.github.kotlinrl.core.agent.Transition
import io.github.kotlinrl.core.algorithm.TransitionLearningAlgorithm
import io.github.kotlinrl.core.api.ParameterSchedule
import io.github.kotlinrl.core.api.Policy
import io.github.kotlinrl.core.api.PolicyUpdate
import io.github.kotlinrl.tabular.QTable
import io.github.kotlinrl.tabular.QTableUpdate
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.random.Random

class ExpectedSARSA(
    initialPolicy: Policy<Int, Int>,
    onPolicyUpdate: PolicyUpdate<Int, Int>,
    rng: Random,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TransitionLearningAlgorithm<Int, Int>(initialPolicy, onPolicyUpdate, rng)  {
    override fun observe(transition: Transition<Int, Int>) {
        val (state, action, reward, sPrime, _, _, isTerminal) = transition
        val alpha = alpha().current
        val distribution = policy[sPrime]
        val expectedQ = if(!isTerminal) {
            distribution.support().map { it to distribution[it] }.sumOf { (aPrime, prob) ->
                prob * Q[sPrime, aPrime]
            }
        } else 0.0

        Q[state, action] = Q[state, action] + alpha * (reward + gamma * expectedQ - Q[state, action])
        onQUpdate(Q)
    }
}