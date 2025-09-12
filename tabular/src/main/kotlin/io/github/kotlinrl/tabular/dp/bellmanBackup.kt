package io.github.kotlinrl.tabular.dp

import org.jetbrains.kotlinx.multik.api.linalg.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Computes state value updates using the Bellman backup operator in the context of reinforcement learning.
 * The Bellman backup computes the updated value for a given state by considering the immediate reward
 * from a state-action pair and the expected discounted future rewards from successor states.
 */
object bellmanBackup {
    /**
     * Performs a Bellman backup computation which calculates the updated
     * value for a given state based on the reward function, transition probabilities,
     * a discount factor, and the value function.
     *
     * R(s,a) + γ∑s′∈S P(s′|s,a)V(s′)
     *
     * @param state The current state for which the backup is performed.
     * @param action The action taken from the current state.
     * @param gamma The discount factor, which determines the importance of future rewards.
     * @param R The reward function that maps a state-action pair to a reward value.
     * @param T The transition function that returns the probability distribution
     *          over future states given a state-action pair.
     * @param V The value function that provides a mapping from states to their estimated values.
     * @return The updated value of the current state after applying the Bellman backup.
     */
    operator fun invoke(
        state: Int,
        action: Int,
        R: D2Array<Double>,
        T: D3Array<Double>,
        gamma: Double,
        V: D1Array<Double>
    ): Double = R[state, action] + gamma * (T[state, action] dot V)
}