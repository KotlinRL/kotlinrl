package io.github.kotlinrl.core.traces

import io.github.kotlinrl.core.*


/**
 * Represents an eligibility trace in reinforcement learning, used to assign credit
 * to past state-action pairs during learning. Eligibility traces are commonly employed
 * in temporal-difference learning algorithms such as SARSA(λ) and Q(λ).
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions in the environment.
 */
interface EligibilityTrace<State, Action> {
    /**
     * Updates the eligibility trace by setting the trace value of the specified state-action pair to 1.0.
     * This method is typically called when a particular state-action pair is visited during a learning process.
     *
     * @param state The state associated with the state-action pair being updated.
     * @param action The action associated with the state-action pair being updated.
     * @return The updated eligibility trace with the specified state-action pair's trace value set to 1.0.
     */
    fun update(state: State, action: Action): EligibilityTrace<State, Action>
    /**
     * Applies the decay operation to the eligibility trace by scaling each trace value
     * by the product of the specified discount factor (gamma) and decay factor (lambda).
     *
     * @param gamma The discount factor that determines the importance of future rewards.
     * @param lambda The decay factor that influences the reduction of trace values over time.
     * @return The updated eligibility trace after the decay operation.
     */
    fun decay(gamma: Double, lambda: Double): EligibilityTrace<State, Action>
    /**
     * Returns a map of state-action keys to their corresponding trace values.
     * The map represents the current state of the eligibility trace, where each entry
     * consists of a state-action pair as the key and its associated trace value.
     *
     * @return A map containing state-action keys and their respective trace values.
     */
    fun values(): Map<StateActionKey<State, Action>, Double>
    /**
     * Clears all state-action trace values by removing all entries from the eligibility trace.
     *
     * @return The eligibility trace after all entries have been removed.
     */
    fun clear(): EligibilityTrace<State, Action>
}