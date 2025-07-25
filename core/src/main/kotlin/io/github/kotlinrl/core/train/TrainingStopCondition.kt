package io.github.kotlinrl.core.train

fun interface TrainingStopCondition {
    operator fun invoke(result: TrainingResult): Boolean
}