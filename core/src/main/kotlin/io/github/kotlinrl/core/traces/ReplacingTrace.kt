package io.github.kotlinrl.core.traces

import io.github.kotlinrl.core.*

/**
 * A concrete implementation of `EligibilityTrace` where eligibility traces are managed
 * using a replacing strategy. For every update, the trace value for the specified state-action
 * pair is set to 1.0, which replaces any previous value. This implementation stores the
 * eligibility traces in a mutable map, indexed by `StateActionKey`.
 *
 * This class is commonly used in temporal-difference learning algorithms like SARSA(λ) and Q(λ)
 * to manage the assignment of credit to previously visited state-action pairs.
 *
 * @param State The type representing states in the environment.
 * @param Action The type representing actions in the environment.
 */
class ReplacingTrace<State, Action> : EligibilityTrace<State, Action> {
    private val traces = mutableMapOf<StateActionKey<State, Action>, Double>()

    /**
     * Updates the eligibility trace for the given state-action pair by setting its trace value to 1.0.
     * Any existing value for the specified state-action pair is replaced with 1.0, in line with
     * the replacing trace strategy.
     *
     * @param state The state associated with the state-action pair to be updated.
     * @param action The action associated with the state-action pair to be updated.
     * @return The updated eligibility trace with the trace value for the specified state-action pair set to 1.0.
     */
    override fun update(state: State, action: Action): EligibilityTrace<State, Action> {
        val key = StateActionKey(state, action)
        traces[key] = 1.0
        return this
    }

    /**
     * Applies the decay operation to all eligibility traces by scaling each trace value
     * by the product of the discount factor (`gamma`) and the decay factor (`lambda`).
     *
     * @param gamma The discount factor that determines the importance of future rewards.
     * @param lambda The decay factor that influences how quickly trace values decay over time.
     * @return The updated eligibility trace after the decay operation is applied.
     */
    override fun decay(gamma: Double, lambda: Double): EligibilityTrace<State, Action> {
        traces.replaceAll { _, value -> gamma * lambda * value }
        return this
    }

    /**
     * Returns a map of state-action keys to their corresponding trace values.
     * This method provides the current state of the eligibility trace,
     * where keys represent state-action pairs, and values represent their eligibility trace values.
     *
     * @return A map containing state-action keys and their respective trace values.
     */
    override fun values(): Map<StateActionKey<State, Action>, Double> = traces.toMap()


    /**
     * Clears all state-action trace values by removing all entries from the internal data structure.
     *
     * @return The eligibility trace after clearing all entries.
     */
    override fun clear(): EligibilityTrace<State, Action> {
        traces.clear()
        return this
    }
}
