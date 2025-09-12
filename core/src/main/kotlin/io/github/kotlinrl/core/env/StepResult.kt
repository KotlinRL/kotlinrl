package io.github.kotlinrl.core.env

/**
 * Represents the result of a single step in an environment.
 *
 * This data class encapsulates the outcome of executing an action within the environment,
 * including the resulting state, any reward obtained, flags indicating whether the episode
 * has terminated or been truncated, and any additional metadata.
 *
 * @param State The type representing the state of the environment.
 * @property state The resulting state of the environment after the action is performed.
 * @property reward The reward obtained from performing the action.
 * @property terminated A flag indicating whether the episode has terminated (e.g., due to reaching a goal or a failure condition).
 * @property truncated A flag indicating whether the episode has been truncated (e.g., due to reaching a time limit or other constraints).
 * @property info A map containing additional metadata or information about the step result.
 */
data class StepResult<State>(
    val state: State,
    val reward: Double,
    val terminated: Boolean,
    val truncated: Boolean,
    val info: Map<String, Any?> = emptyMap()
)