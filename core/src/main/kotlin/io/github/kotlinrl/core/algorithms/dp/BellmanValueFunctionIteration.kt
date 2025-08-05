package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class BellmanValueFunctionIteration<State, Action>(
    private var initialV: EnumerableValueFunction<State>,
    private var model: MDPModel<State, Action>,
    private var gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActions: StateActions<State, Action>,
    private val onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
) : DPIteration<State, Action>() {

    override fun plan(): Policy<State, Action> {
        var delta: Double
        var V = initialV

        do {
            delta = 0.0
            var newV = V

            for (s in model.allStates()) {
                val actions = stateActions(s)
                if (actions.isEmpty()) continue

                val bestActionValue = actions.maxOf { a -> expectedReturn(s, a, V) }
                delta = maxOf(delta, abs(bestActionValue - V[s]))
                newV = newV.update(s, bestActionValue)
            }

            V = newV
            onValueFunctionUpdate(V)

        } while (delta > theta)

        return Policy { s ->
            val actions = stateActions(s)
            actions.maxByOrNull { a -> expectedReturn(s, a, V) } ?: actions.random()
        }
    }

    private fun expectedReturn(s: State, a: Action, V: EnumerableValueFunction<State>): Double {
        return model.transitions(s, a).sumOf { t ->
            t.probability * (t.reward + gamma * if (t.done) 0.0 else V[t.nextState])
        }
    }
}
