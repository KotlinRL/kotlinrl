package io.github.kotlinrl.core.env

/**
 * Represents the initial state of an environment and includes auxiliary metadata.
 *
 * This data class is typically used to encapsulate the state of an environment
 * upon initialization or reset. It allows for carrying both the state information
 * and additional metadata, which can provide context or configurational details
 * relevant to the current environment setup.
 *
 * @param State The type representing the state of the environment.
 * @property state The initial state of the environment.
 * @property info A map containing additional metadata or information about the environment.
 */
data class InitialState<State>(
    val state: State,
    val info: Map<String, Any> = mapOf()
)
