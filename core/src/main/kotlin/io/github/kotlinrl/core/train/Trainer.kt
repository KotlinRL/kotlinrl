package io.github.kotlinrl.core.train

interface Trainer {
    fun train(episodes: Int): TrainingResult
}