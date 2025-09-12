package io.github.kotlinrl.core.model

import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Represents a Markov Decision Process (MDP), a foundational framework in reinforcement learning
 * and decision-making under uncertainty. An MDP is defined by its states, actions, reward function,
 * transition probabilities, and a discount factor.
 *
 * @property S A list of all possible states in the MDP.
 * @property A A functional interface defining the actions available for each state.
 * @property R The reward function, which computes the reward for state-action pairs.
 * @property T The transition function, which defines the probability distribution of next states
 * for a given state-action pair.
 * @property gamma The discount factor, used to weigh immediate rewards against future rewards.
 * Expected to have a value between 0 (only immediate rewards are considered) and 1 (future rewards
 * are fully considered).
 * @property isTerminal A predicate that determines whether a given state is terminal. Terminal states
 * are those where no further actions or transitions are possible.
 */
data class TabularMDP(
    override val S: FiniteStates,
    override val A: FixedIntActions,
    val RA: D2Array<Double>,
    val TA: D3Array<Double>,
    override val gamma: Double,
) : FiniteTabular {
    override val R = { state: Int, action: Int -> RA[state, action] }
    override val T = { state: Int, action: Int, sPrime: Int -> TA[state, action, sPrime] }
}