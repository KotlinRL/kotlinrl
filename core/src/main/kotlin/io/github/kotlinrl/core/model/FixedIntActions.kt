package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.api.*

data class FixedIntActions(val numActions: Int) : Actions<Int, Int> {
    override fun get(state: Int): List<Int> = (0 until numActions).toList()
}
