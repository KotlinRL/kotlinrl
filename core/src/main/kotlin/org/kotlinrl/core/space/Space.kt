package org.kotlinrl.core.space

import kotlin.random.Random

interface Space<T> {
    val random: Random
    fun sample(): T
    fun contains(value: T): Boolean
}