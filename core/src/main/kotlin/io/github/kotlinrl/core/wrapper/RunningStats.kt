package io.github.kotlinrl.core.wrapper

/**
 * A utility class for tracking and calculating running statistics such as mean and standard deviation
 * in a numerically stable manner.
 *
 * The class maintains an internal state which is updated with each new value using the `update` function.
 */
class RunningStats {
    var mean = 0.0
    var varSum = 0.0
    var count = 0

    /**
     * Updates the internal statistics with a new data point.
     *
     * This method adjusts the mean, count, and variance sum (varSum) incrementally
     * to include the new value, ensuring numerical stability.
     *
     * @param x The new data point to update the statistics with.
     */
    fun update(x: Double) {
        count += 1
        val delta = x - mean
        mean += delta / count
        varSum += delta * (x - mean)
    }

    /**
     * Computes the standard deviation of the data points tracked by the `RunningStats` class.
     *
     * The standard deviation is calculated using the current variance sum (`varSum`) and the number
     * of data points (`count`). This implementation ensures numerical stability, particularly
     * for cases with incremental updates. If the number of data points is less than 2, the standard
     * deviation is returned as a small constant (`1e-8`) to avoid division by zero.
     *
     * This is a derived property and does not store its value directly; instead, it dynamically calculates
     * the standard deviation whenever accessed.
     *
     * @return The standard deviation of the tracked data points.
     */
    val std: Double
        get() = if (count > 1) kotlin.math.sqrt(varSum / (count - 1)) else 1e-8
}