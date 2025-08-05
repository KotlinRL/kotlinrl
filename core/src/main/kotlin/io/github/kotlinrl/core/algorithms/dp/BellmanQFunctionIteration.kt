package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class BellmanQFunctionIteration<State, Action>(
    private val initialQ: EnumerableQFunction<State, Action>,
    private val model: MDPModel<State, Action>,
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActions: StateActions<State, Action>,
    private val onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
) : DPIteration<State, Action>() {

    override fun plan(): Policy<State, Action> {
        var delta: Double
        var Q = initialQ

        do {
            delta = 0.0
            var newQ = Q

            for (s in model.allStates()) {
                val actions = stateActions(s)
                for (a in actions) {
                    val transitions = model.transitions(s, a)
                    val expectedValue = transitions.sumOf { t ->
                        val maxQNext = if (t.done) 0.0 else {
                            val nextActions = stateActions(t.nextState)
                            nextActions.maxOfOrNull { Q[t.nextState, it] } ?: 0.0
                        }
                        t.probability * (t.reward + gamma * maxQNext)
                    }

                    delta = maxOf(delta, kotlin.math.abs(Q[s, a] - expectedValue))
                    newQ = newQ.update(s, a, expectedValue)
                }
            }

            Q = newQ
            onQFunctionUpdate(Q)
        } while (delta > theta)

        return Policy { s ->
            val actions = stateActions(s)
            actions.maxByOrNull { a -> Q[s, a] } ?: actions.random()
        }
    }
}
