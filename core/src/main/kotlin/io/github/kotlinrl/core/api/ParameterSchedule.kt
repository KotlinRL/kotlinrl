package io.github.kotlinrl.core.api

import io.github.kotlinrl.core.ParameterScheduleDecay

/**
 * Represents a parameter schedule used to dynamically adjust a specific parameter
 * within reinforcement learning algorithms. The schedule is defined as a function
 * that, when invoked, provides a `Double` value representing the current parameter setting.
 *
 * This interface is commonly used to manage parameters such as exploration rates (e.g., epsilon in epsilon-greedy policies)
 * or temperature in softmax policies, allowing these parameters to change over time or depend on the learning process.
 */
fun interface ParameterSchedule {
    companion object {
        /**
         * Creates a constant parameter schedule that always returns the given value.
         *
         * This function generates a `ParameterSchedule` where the parameter values
         * (previous, current, and minimum value) are all set to the provided constant value.
         * It is useful when a static, unchanging parameter value is required.
         *
         * @param value The constant value to be used for the parameter schedule.
         * @return A `ParameterSchedule` that evaluates to the specified constant value.
         */
        fun constant(value: Double): ParameterSchedule = ParameterSchedule { Parameter(value, value, value) }

        /**
         * Creates a linear decay schedule for a parameter, where the value decreases linearly
         * over time until it reaches a specified minimum value, with an optional burn-in period.
         *
         * @param initialValue The initial value of the parameter at the start of the schedule.
         * @param decayRate The rate at which the parameter value decreases at each step.
         * @param minValue The minimum value the parameter can reach during the decay process.
         * @param burnInEpisodes The number of steps to delay the start of the decay process (default is 0).
         * @param callback A function to be invoked after each decay step, receiving the current step number and parameter state.
         * @return A pair consisting of a `ParameterSchedule` (to query the current parameter state)
         *         and a `ParameterScheduleDecay` (to perform the decay step).
         */
        fun linearDecay(
            initialValue: Double,
            decayRate: Double,
            minValue: Double,
            burnInEpisodes: Int = 0,
            callback: (Int, Parameter) -> Unit = { _, _ -> }
        ): Pair<ParameterSchedule, ParameterScheduleDecay> {

            var decayStep = 0
            var previous = initialValue
            var current = initialValue

            val schedule = ParameterSchedule {
                Parameter(previous, current, minValue)
            }

            val decay: ParameterScheduleDecay = {
                previous = current
                if (decayStep >= burnInEpisodes) {
                    current = (current - decayRate).coerceAtLeast(minValue)
                }
                decayStep++
                callback(decayStep, Parameter(current, previous, minValue))
            }

            return schedule to decay
        }

        /**
         * Creates a geometric decay schedule for parameter adjustment, commonly used in optimization or reinforcement learning.
         * This function computes a parameter schedule and a decay mechanism based on the initial value, decay rate,
         * minimum value, and an optional burn-in period. A callback function is provided to monitor decay step changes.
         *
         * @param initialValue the initial value of the parameter before decay begins.
         * @param decayRate the rate of decay applied to the parameter in each step.
         * @param minValue the minimum allowable value for the parameter during decay.
         * @param burnInEpisodes the number of steps to skip before starting decay (default is 0).
         * @param callback a function invoked on each decay step with the current step number and the parameter state (default is a no-op).
         * @return a pair of `ParameterSchedule` and `ParameterScheduleDecay` for managing and applying the decay schedule.
         */
        fun geometricDecay(
            initialValue: Double,
            decayRate: Double,
            minValue: Double,
            burnInEpisodes: Int = 0,
            callback: (Int, Parameter) -> Unit = { _, _ -> }
        ): Pair<ParameterSchedule, ParameterScheduleDecay> {

            var decayStep = 0
            var previous = initialValue
            var current = initialValue

            val schedule = ParameterSchedule {
                Parameter(previous, current, minValue)
            }

            val decay: ParameterScheduleDecay = {
                previous = current
                if (decayStep >= burnInEpisodes) {
                    current = (current * decayRate).coerceAtLeast(minValue)
                }
                decayStep++
                callback(decayStep, Parameter(current, previous, minValue))
            }

            return schedule to decay
        }
    }

    /**
     * Computes and returns the current parameter value defined by the schedule.
     *
     * This method represents the evaluation of the parameter schedule, providing
     * a dynamically adjusted value based on the implementation of the schedule,
     * such as time or iteration-dependent adjustments.
     *
     * @return the current parameter value as a Double.
     */
    operator fun invoke(): Parameter
}