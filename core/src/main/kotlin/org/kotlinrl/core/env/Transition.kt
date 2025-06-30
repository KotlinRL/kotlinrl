package org.kotlinrl.core.env

data class Transition<Observation, Reward>(
    val observation: Observation,
    val reward: Reward,
    val terminated: Boolean,
    val truncated: Boolean,
    val info: Map<String, String>
)