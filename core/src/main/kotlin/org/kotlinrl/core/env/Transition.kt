package org.kotlinrl.core.env

data class Transition<Observation>(
    val observation: Observation,
    val reward: Double,
    val terminated: Boolean,
    val truncated: Boolean,
    val info: Map<String, String>
)