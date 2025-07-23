package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*
import kotlin.math.*

class NStepSARSA<State, Action>(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double,
    private val n: Int,
    private val policyProbabilities: PolicyProbabilities<State, Action>
) : TabularTDLearning<State, Action>(qTable, alpha, gamma), TrajectoryLearner<State, Action> {

    private val queue = ArrayDeque<Transition<State, Action>>(n)

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        TODO("Not yet implemented")
    }
//
//    override fun onEpisodeStart(episode: Int) {
//        queue.clear()
//    }
//
//    override fun onEpisodeEnd(stats: EpisodeStats<State, Action>) {
//        while (queue.isNotEmpty<Transition<State, Action>>()) {
//            update(queue.size)
//        }
//    }

    override fun invoke(transition: Transition<State, Action>) {
        queue.addLast(transition)
        if (queue.size >= n) {
            update(n)
        }
    }

    private fun update(steps: Int) {
        val transitions = queue.take(steps)
        val (s0, a0) = transitions.first().state to transitions.first().action

        var g = 0.0
        for (i in transitions.indices) {
            g += gamma.pow(i) * transitions[i].reward
        }

        val last = transitions.last()
        if (!last.done) {
            val expectedQ = policyProbabilities(last.state).entries.sumOf { (a, prob) ->
                prob * qTable[last.state, a]
            }
            g += gamma.pow(transitions.size) * expectedQ
        }

        val currentQ = qTable[s0, a0]
        qTable[s0, a0] = currentQ + alpha * (g - currentQ)

        queue.removeFirst()
    }

}
