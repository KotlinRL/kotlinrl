package io.github.kotlinrl.core

/**
 * A type alias for `io.github.kotlinrl.core.traces.EligibilityTrace`, representing a trace
 * mechanism used in reinforcement learning algorithms to track state-action pairs over time.
 *
 * This alias simplifies reference to `EligibilityTrace`, which is implemented as an interface
 * providing methods for updating traces, decaying them based on parameters like gamma and lambda,
 * retrieving trace values as a map, and clearing the trace.
 *
 * @param State The type representing the states of the environment.
 * @param Action The type representing the actions available in the environment.
 */
typealias EligibilityTrace<State, Action> = io.github.kotlinrl.core.traces.EligibilityTrace<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.traces.ReplacingTrace`.
 *
 * Represents an implementation of eligibility traces using replacing methodology in reinforcement learning.
 * The trace updates its value to 1 when the state-action pair is visited and decays over time
 * based on the decay factors (gamma and lambda) provided.
 *
 * Provides functionalities for updating trace values, decaying existing trace values,
 * clearing the stored traces, and accessing the current set of eligibility trace values.
 *
 * @param State The type representing the state in the trace.
 * @param Action The type representing the action in the trace.
 */
typealias ReplacingTrace<State, Action> = io.github.kotlinrl.core.traces.ReplacingTrace<State, Action>