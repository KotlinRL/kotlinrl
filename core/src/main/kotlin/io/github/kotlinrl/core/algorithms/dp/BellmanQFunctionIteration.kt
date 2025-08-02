package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class BellmanQFunctionIteration<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val model: MDPModel<State, Action>,
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val onQFunctionUpdate: (EnumerableQFunction<State, Action>) -> Unit = { },
) : DPIteration<State, Action>() {

    private var q = initialQ

    override fun plan(): Policy<State, Action> {
        var delta: Double
        var iterations = 0

        do {
            delta = 0.0
            var newQ = q

            for (s in model.allStates()) {
                val actions = stateActionListProvider(s)
                for (a in actions) {
                    val transitions = model.transitions(s, a)
                    val expectedValue = transitions.sumOf { t ->
                        val maxQNext = if (t.done) 0.0 else {
                            val nextActions = stateActionListProvider(t.nextState)
                            nextActions.maxOfOrNull { q[t.nextState, it] } ?: 0.0
                        }
                        t.probability * (t.reward + gamma * maxQNext)
                    }

                    delta = maxOf(delta, kotlin.math.abs(q[s, a] - expectedValue))
                    newQ = newQ.update(s, a, expectedValue)
                }
            }

            q = newQ
            onQFunctionUpdate(q)
            iterations++
        } while (delta > theta)

        return Policy { s ->
            val actions = stateActionListProvider(s)
            actions.maxByOrNull { a -> q[s, a] } ?: actions.random()
        }
    }
}
