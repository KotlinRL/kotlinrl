package io.github.kotlinrl.core.policy

interface StochasticPolicy<State, Action> : Policy<State, Action> {
    fun probability(state: State, action: Action): Double
}
