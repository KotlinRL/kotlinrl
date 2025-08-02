package io.github.kotlinrl.core.policy

fun interface PolicyImprovementStrategy<State, Action> {
    operator fun invoke(q: QFunction<State, Action>): Policy<State, Action>
}
