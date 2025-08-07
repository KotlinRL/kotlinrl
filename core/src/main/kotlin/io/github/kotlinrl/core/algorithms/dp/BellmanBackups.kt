package io.github.kotlinrl.core.algorithms.dp

object BellmanBackups {
    fun <State, Action> default(): BellmanBackup<State, Action> =
        BellmanBackup { r, next, p, done ->
            p * (r + if (done) 0.0 else next)
        }

    fun <State, Action> discounted(gamma: Double): BellmanBackup<State, Action> =
        BellmanBackup { r, next, p, done ->
            p * (r + gamma * if (done) 0.0 else next)
        }
}