package io.github.kotlinrl.core.policy

fun interface ProbabilityFunction<State, Action> {
    operator fun invoke(state: State, action: Action): Double
}
