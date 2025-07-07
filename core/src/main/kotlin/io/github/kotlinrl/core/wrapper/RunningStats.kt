package io.github.kotlinrl.core.wrapper

class RunningStats {
    var mean = 0.0
    var varSum = 0.0
    var count = 0

    fun update(x: Double) {
        count += 1
        val delta = x - mean
        mean += delta / count
        varSum += delta * (x - mean)
    }

    val std: Double
        get() = if (count > 1) kotlin.math.sqrt(varSum / (count - 1)) else 1e-8
}