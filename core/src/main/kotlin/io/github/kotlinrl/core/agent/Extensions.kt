@file:JvmName("AgentKt")

package io.github.kotlinrl.core.agent

/**
 * Represents a trajectory in a reinforcement learning context, which is a sequence of transitions
 * observed during an episode of interaction between an agent and the environment.
 *
 * A trajectory comprises a list of transitions, where each transition contains information about
 * the state, action, reward, next state, and termination or truncation status.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias Trajectory<State, Action> = List<Transition<State, Action>>