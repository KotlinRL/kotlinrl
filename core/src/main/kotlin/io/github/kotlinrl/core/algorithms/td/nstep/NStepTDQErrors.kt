package io.github.kotlinrl.core.algorithms.td.nstep

import kotlin.math.*

object NStepTDQErrors {
    @Suppress("DuplicatedCode")
    fun <State, Action> nStepExpectedSARSA(): NStepTDQError<State, Action> =
        NStepTDQError { Q, traj, policy, _, gamma ->
            require(policy != null) { "Policy required for n-step Expected SARSA." }
            val (s0, a0) = traj.first().state to traj.first().action

            var G = 0.0
            traj.forEachIndexed { i, t -> G += gamma.pow(i) * t.reward }

            val last = traj.last()
            if (!last.done) {
                val sT = last.nextState                  // <-- bootstrap at S_{t+n}
                val exp = policy.probabilities(sT).entries
                    .sumOf { (a, p) -> p * Q[sT, a] }
                G += gamma.pow(traj.size) * exp
            }

            G - Q[s0, a0]
        }

    // n-step SARSA (on-policy): bootstrap with the action actually taken at time t+n
    @Suppress("DuplicatedCode")
    fun <State, Action> nStepSARSA(): NStepTDQError<State, Action> =
        NStepTDQError { Q, traj, _, tailAction, gamma ->
            val (s0, a0) = traj.first().state to traj.first().action

            val n = traj.size
            var G = 0.0
            for (i in 0 until n) {
                G += gamma.pow(i) * traj[i].reward
            }

            val last = traj.last()
            if (!last.done && tailAction != null) {
                val sT = last.nextState
                G += gamma.pow(n) * Q[sT, tailAction]
            }

            G - Q[s0, a0]
        }

    // n-step Q-learning (off-policy): bootstrap with max_a Q(S_{t+n}, a)
    @Suppress("DuplicatedCode")
    fun <State, Action> nStepQLearning(): NStepTDQError<State, Action> =
        NStepTDQError { Q, traj, _, _, gamma ->
            val (s0, a0) = traj.first().state to traj.first().action

            var G = 0.0
            traj.forEachIndexed { i, t -> G += gamma.pow(i) * t.reward }

            val last = traj.last()
            if (!last.done) {
                val sT = last.nextState
                G += gamma.pow(traj.size) * Q.maxValue(sT)
            }

            G - Q[s0, a0]
        }
}