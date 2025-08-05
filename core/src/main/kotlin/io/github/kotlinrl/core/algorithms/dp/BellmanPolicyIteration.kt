package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class BellmanPolicyIteration<State, Action>(
    private var initialPolicy: Policy<State, Action>,
    private var initialV: EnumerableValueFunction<State>,
    private val model: MDPModel<State, Action>,
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    private val stateActions: StateActions<State, Action>,
    private val onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
    private val onPolicyUpdate: PolicyUpdate<State, Action> = { }
) : DPIteration<State, Action>() {

    private val estimator = BellmanValueFunctionEstimator<State, Action>(gamma)

    override fun plan(): Policy<State, Action> {
        var policy = initialPolicy
        var V = initialV
        var stable: Boolean

        do {
            var delta: Double
            do {
                delta = 0.0
                val transitions = model.allStates().flatMap { s ->
                    model.transitions(s, policy(s))
                }

                val updatedV = estimator.estimate(V, transitions)

                for (s in updatedV.allStates()) {
                    delta = maxOf(delta, abs(updatedV[s] - V[s]))
                }

                V = updatedV
                onValueFunctionUpdate(V)
            } while (delta > theta)

            stable = true
            val newPolicy = Policy<State, Action> { s ->
                val actions = stateActions(s)
                val best = actions.maxByOrNull { a ->
                    model.transitions(s, a).sumOf { t ->
                        t.probability * (t.reward + gamma * if (t.done) 0.0 else V[t.nextState])
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

