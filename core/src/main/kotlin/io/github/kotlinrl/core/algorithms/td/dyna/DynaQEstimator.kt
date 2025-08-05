package io.github.kotlinrl.core.algorithms.td.dyna

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.td.TDQErrors

class DynaQEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val model: LearnableMDPModel<State, Action>,
    private val planningSteps: Int = 5,
    private val td: TDQError<State, Action> = TDQErrors.qLearning()
) : TransitionQFunctionEstimator<State, Action> {

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a) = transition
        val done = transition.done

        // Real experience Q-learning
        val delta = td(Q, transition, null, gamma, done)
        var currentQ = Q.update(s, a, Q[s, a] + alpha() * delta)

        // Update model
        model.update(transition)

        // Planning phase
        repeat(planningSteps) {
            val sampleTransition = model.sampleTransition() ?: return@repeat
            val (sPlan, aPlan) = sampleTransition
            val done = sampleTransition.done
            val deltaPlan = td(currentQ, sampleTransition, null, gamma, done)
            currentQ = currentQ.update(sPlan, aPlan, currentQ[sPlan, aPlan] + alpha() * deltaPlan)
        }
        return currentQ
    }
}