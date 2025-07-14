package io.github.kotlinrl.core.plan

fun interface StateProvider<State> {
    operator fun invoke(): Iterable<State>
}