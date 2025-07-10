package io.github.kotlinrl.core.policy

fun interface PolicyProbabilities<State, Action> {
    operator fun invoke(state: State): Map<Action, Double>
}