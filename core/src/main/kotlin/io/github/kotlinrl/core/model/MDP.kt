package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.R
import io.github.kotlinrl.core.T
import io.github.kotlinrl.core.api.*

/**
 * Represents a Markov Decision Process (MDP), a fundamental model in reinforcement learning
 * and decision-making under uncertainty. An MDP is defined by its states, actions,
 * reward function, transition probabilities, and a discount factor.
 *
 * @param State The type representing the states in the MDP.
 * @param Action The type representing the actions in the MDP.
 *
 * @property S A collection of possible states in the MDP.
 * @property A A functional interface defining the available actions for each state.
 * @property R The reward function, which computes the reward value for a given state-action pair.
 * @property T The transition function, which defines the probability distribution over next states
 * for a given state-action pair.
 * @property gamma The discount factor that determines the weight of future rewards compared to immediate rewards.
 * Values typically range between 0 (only immediate rewards matter) and 1 (future rewards are fully considered).
 */
interface MDP<State, Action> {
    val S: States<State>
    val A: Actions<State, Action>
    val R: R<State, Action>
    val T: T<State, Action>
    val gamma: Double
}