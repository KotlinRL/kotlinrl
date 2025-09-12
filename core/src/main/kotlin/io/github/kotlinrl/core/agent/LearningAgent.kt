package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.algorithm.LearningAlgorithm

/**
 * Represents a learning agent capable of interacting with an environment
 * and adapting its behavior based on feedback using a specified learning algorithm.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @property id The unique identifier of the learning agent.
 * @property algorithm The learning algorithm used by the agent to determine actions and update its behavior.
 */
class LearningAgent<State, Action>(
    override val id: String,
    val algorithm: LearningAlgorithm<State, Action>,
) : Agent<State, Action> {

    /**
     * Determines the action to take given the current state.
     *
     * This method leverages the underlying learning algorithm to process the provided state
     * and determine an appropriate action based on its policy or decision-making logic.
     *
     * @param state The current state of the environment.
     * @return The action chosen by the agent based on the provided state.
     */
    override fun act(state: State): Action = algorithm(state)

    /**
     * Observes a single transition in the environment, typically for the purpose of learning or updating the agent's behavior.
     *
     * This method processes the observed transition, which includes the current state, action taken, resulting reward,
     * next state, and whether the transition marks the termination or truncation of the environment.
     * The transition is relayed to the underlying learning algorithm for updates.
     *
     * @param transition The transition object containing details of the observed state, chosen action, resulting reward,
     * next state, and termination or truncation flags.
     */
    override fun observe(transition: Transition<State, Action>) = algorithm.update(transition)

    /**
     * Observes a trajectory of transitions across a specific episode for learning or updating the agent's behavior.
     *
     * This method delegates the processing of the trajectory and episode information to the underlying
     * learning algorithm, which is responsible for handling updates or adjustments based on the provided data.
     *
     * @param trajectory The sequence of transitions observed during the episode, encapsulating
     * states, actions, rewards, and additional metadata.
     * @param episode The identifier or index of the current episode being observed.
     */
    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) = algorithm.update(trajectory, episode)
}
