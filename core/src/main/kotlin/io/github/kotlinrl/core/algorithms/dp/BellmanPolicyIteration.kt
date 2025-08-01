package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class BellmanPolicyIteration<State, Action>(
    private var initialPolicy: Policy<State, Action>,
    private var initialV: EnumerableValueFunction<State>,
    private val model: MDPModel<State, Action>,
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val onValueFunctionUpdate: (EnumerableValueFunction<State>) -> Unit = { },
    private val onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
) : DPIteration<State, Action>() {

    private val estimator = BellmanValueFunctionEstimator<State, Action>(gamma)

    override fun plan(): Policy<State, Action> {
        var policy = initialPolicy
        var v = initialV
        var stable: Boolean
        var iterations = 0

        do {
            iterations++

            var delta: Double
            do {
                delta = 0.0
                val transitions = model.allStates().flatMap { s ->
                    model.transitions(s, policy(s))
                }

                val updatedV = estimator.estimate(v, transitions)

                for (s in updatedV.allStates()) {
                    delta = maxOf(delta, abs(updatedV[s] - v[s]))
                }

                v = updatedV
                onValueFunctionUpdate(v)
            } while (delta > theta)

            stable = true
            val newPolicy = Policy<State, Action> { s ->
                val actions = stateActionListProvider(s)
                val best = actions.maxByOrNull { a ->
                    model.transitions(s, a).sumOf { t ->
                        t.probability * (t.reward + gamma * if (t.done) 0.0 else v[t.nextState])
                    }
                }
                best ?: actions.random()
            }

            for (s in model.allStates()) {
                if (policy(s) != newPolicy(s)) {
                    stable = false
                    break
                }
            }

            policy = newPolicy
            onPolicyUpdate(policy)

        } while (!stable)

        return policy
    }
}

