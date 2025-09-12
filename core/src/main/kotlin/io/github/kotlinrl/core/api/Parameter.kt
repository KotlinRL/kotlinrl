package io.github.kotlinrl.core.api

/**
 * Represents a parameter state with values for tracking progression or adjustments
 * during iterative processes, such as parameter scheduling in algorithms.
 *
 * The `Parameter` class is typically used to encapsulate a parameter's current value,
 * its previous value, and an enforced minimum value. It is primarily utilized in dynamic
 * parameter management scenarios, such as reinforcement learning or optimization tasks.
 *
 * @property previous The parameter value from the previous iteration or step.
 * @property current The current parameter value at this stage of the process.
 * @property minValue The minimum allowed value for the parameter, ensuring it does not fall below this threshold.
 */
data class Parameter(
    val previous: Double,
    val current: Double,
    val minValue: Double,
)
