package io.github.kotlinrl.core.api

/**
 * Represents a parameter configuration with current and previous values,
 * alongside constraints and decay characteristic.
 *
 * This data class is designed to be used in scenarios where tracking and updating
 * parameters over time is necessary, such as reinforcement learning or optimization processes.
 *
 * @property current The current value of the parameter.
 * @property previous The previous value the parameter held.
 * @property minValue The minimum allowed value for the parameter as a constraint.
 * @property decayStep The step at which the parameter decays, typically indicative of iterations or timeframes.
 */
data class Parameter(
    val current: Double,
    val previous: Double,
    val minValue: Double,
    val decayStep: Int
)
