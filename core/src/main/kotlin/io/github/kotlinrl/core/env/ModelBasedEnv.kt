package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.*

interface ModelBasedEnv : Env<IntArray, Int, MultiDiscrete, Discrete> {

    fun simulateStep(state: IntArray, action: Int): Transition<IntArray>
}