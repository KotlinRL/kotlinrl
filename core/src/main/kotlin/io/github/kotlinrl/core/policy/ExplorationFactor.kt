package io.github.kotlinrl.core.policy

fun interface ExplorationFactor {
    operator fun invoke(): Double
}