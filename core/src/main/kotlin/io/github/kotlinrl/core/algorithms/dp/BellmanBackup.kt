package io.github.kotlinrl.core.algorithms.dp


/**
 * Represents a functional interface for performing Bellman backups in reinforcement learning.
 *
 * A Bellman backup is a key step in many dynamic programming and reinforcement learning
 * algorithms, used to compute updates to an estimate of the value or action-value function.
 * It combines the immediate reward, the estimated value of the next state, the transition
 * probability, and terminal state considerations to calculate the updated value.
 *
 * This interface allows customization of how the Bellman backup is defined, enabling
 * implementations such as standard or discounted formulations.
 *
 * @param State The type representing states in the environment or Markov Decision Process (MDP).
 * @param Action The type representing actions in the environment or Markov Decision Process (MDP).
 */
fun interface BellmanBackup<State, Action>  {
    /**
     * Computes an updated value based on the Bellman backup equation, taking into account
     * the immediate reward, the estimated value of the next state, the transition probability,
     * and whether the state is terminal.
     *
     * @param reward The immediate reward received after taking an action in the current state.
     * @param nextStateValue The estimated value of the next state after the transition.
     * @param probability The probability of transitioning to the next state.
     * @param isTerminal A boolean flag indicating whether the next state is terminal.
     * @return The computed updated value for the current state-action pair.
     */
    operator fun invoke(
        reward: Double,
        nextStateValue: Double,
        probability: Double,
        isTerminal: Boolean
    ): Double
}