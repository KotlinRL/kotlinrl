package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.space.Discrete
import io.github.kotlinrl.core.space.MultiDiscrete

interface ModelBasedEnv : Env<IntArray, Int, MultiDiscrete, Discrete> {
    val size: Int

    val goal: IntArray

    fun nextState(state: IntArray, action: Int): IntArray

    fun computeReward(state: IntArray, action: Int): Double

    fun simulateStep(action: Int): Transition<IntArray>

    fun actionProbabilities(state: IntArray): DoubleArray

    fun stateActionList(state: IntArray): List<Int>
}