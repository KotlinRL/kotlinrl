package io.github.kotlinrl.core.space

import kotlin.random.*

interface Space<T> {
    val random: Random
    fun sample(): T
    fun contains(value: Any?): Boolean
}