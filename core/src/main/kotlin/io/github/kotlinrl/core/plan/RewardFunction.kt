package io.github.kotlinrl.core.plan

fun interface RewardFunction<State, Action> {
    operator fun invoke(state: State, action: Action): Double
}