package io.github.kotlinrl.core.train

interface Trainer {
    fun train(stopCondition: TrainingStopCondition): TrainingResult
}