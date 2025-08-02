package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class BellmanValueFunctionIteration<State, Action>(
    private var initialV: EnumerableValueFunction<State>,
    private var model: MDPModel<State, Action>,
    private var gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val onValueFunctionUpdate: (EnumerableValueFunction<State>) -> Unit = { },
) : DPIteration<State, Action>() {

    override fun plan(): Policy<State, Action> {
        var delta: Double
        var v = initialV
        var iterations = 0

        do {
            delta = 0.0
            var newV = v

            for (s in v.allStates()) {
                val actions = stateActionListProvider(s)
                if (actions.isEmpty()) continue

                val bestActionValue = actions.maxOf { a -> expectedReturn(s, a, v) }
                delta = maxOf(delta, abs(bestActionValue - v[s]))
                newV = newV.update(s, bestActionValue)
            }

            v = newV
            iterations++
            onValueFunctionUpdate(v)

        } while (delta > theta)

        return Policy { s ->
            val actions = stateActionListProvider(s)
            actions.maxByOrNull { a -> expectedReturn(s, a, v) } ?: actions.random()
        }
    }

    private fun expectedReturn(s: State, a: Action, v: EnumerableValueFunction<State>): Double {
        return model.transitions(s, a).sumOf { t ->
            t.probability * (t.reward + gamma * if (t.done) 0.0 else v[t.nextState])
        }
    }
}
