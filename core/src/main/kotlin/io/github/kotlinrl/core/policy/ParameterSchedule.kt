package io.github.kotlinrl.core.policy

/**
 * Represents a parameter schedule used to dynamically adjust a specific parameter
 * within reinforcement learning algorithms. The schedule is defined as a function
 * that, when invoked, provides a `Double` value representing the current parameter setting.
 *
 * This interface is commonly used to manage parameters such as exploration rates (e.g., epsilon in epsilon-greedy policies)
 * or temperature in softmax policies, allowing these parameters to change over time or depend on the learning process.
 */
fun interface ParameterSchedule {
    /**
     * Computes and returns the current parameter value defined by the schedule.
     *
     * This method represents the evaluation of the parameter schedule, providing
     * a dynamically adjusted value based on the implementation of the schedule,
     * such as time or iteration-dependent adjustments.
     *
     * @return the current parameter value as a Double.
     */
    operator fun invoke(): Double
}