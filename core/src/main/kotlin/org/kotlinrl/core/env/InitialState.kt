package org.kotlinrl.core.env

data class InitialState<Observation>(
    val observation: Observation,
    val info: Map<String, Any> = mapOf()
)
