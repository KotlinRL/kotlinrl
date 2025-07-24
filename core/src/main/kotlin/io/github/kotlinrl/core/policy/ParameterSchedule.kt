package io.github.kotlinrl.core.policy

fun interface ParameterSchedule {
    operator fun invoke(): Double
}