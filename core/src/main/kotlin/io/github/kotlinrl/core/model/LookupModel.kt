package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*

/**
 * Represents a lookup-based implementation of a learnable and expandable Markov Decision Process (MDP) model.
 *
 * This model provides mechanisms to:
 * - Store state-action transition probabilities and rewards.
 * - Update the MDP with observed transitions.
 * - Retrieve information about states, actions, transitions, rewards, and terminality based on observations.
 * - Sample transitions from the observed data.
 *
 * @param State The type representing the states in the MDP.
 * @param Action The type representing the actions in the MDP.
 * @constructor Initializes the `LookupModel` with a list of available actions in the MDP.
 *
 * @property actions List of actions available to use in the MDP for any given state.
 */
class LookupModel<State, Action>(
    private val actions: List<Action>
) :LearnableMDPModel<State, Action>, ExpandableMDPModel<State, Action> {

    private val stateSet = mutableSetOf<Comparable<*>>()
    private val terminalSet = mutableSetOf<Comparable<*>>()
    private val transitionCounts = mutableMapOf<StateActionKey<*, *>, MutableMap<Comparable<*>, Int>>()
    private val rewardSums = mutableMapOf<StateActionKey<*, *>, Double>()
    private val visitCounts = mutableMapOf<StateActionKey<*, *>, Int>()
    private val predecessorsMap = mutableMapOf<Comparable<*>, MutableSet<StateActionKey<*, *>>>()

    /**
     * Retrieves a list of all states in the model. If an element in the internal state set is of type `ComparableIntList`,
     * it transforms the element's data into an NDArray representation. Otherwise, the element is returned as-is.
     *
     * @return A list of states, each represented as either their original type or transformed to an NDArray if applicable.
     */
    override fun allStates(): List<State> = stateSet.map {
        @Suppress("UNCHECKED_CAST")
        when(it) {
            is ComparableIntList -> mk.ndarray(it.data).asDNArray()
            else -> it
        } as State
    }.toList()

    /**
     * Retrieves a list of all actions available in the model.
     *
     * @return A list containing all actions represented in the model.
     */
    override fun allActions(): List<Action> = actions

    /**
     * Computes probabilistic transitions for a given state and action.
     * This method uses internal transition and reward data to generate
     * probabilistic outcomes, including the next state, reward, probability of occurrence,
     * and terminal status.
     *
     * @param state The current state from which the action is executed.
     * @param action The action to be performed in the given state.
     * @return A probabilistic trajectory containing transitions with associated probabilities, rewards, and resulting states.
     */
    override fun transitions(state: State, action: Action): ProbabilisticTrajectory<State, Action> {
        val key = stateActionKey(state, action)
        val total = visitCounts[key] ?: return emptyList()
        val outcomes = transitionCounts[key] ?: return emptyList()
        val avgReward = expectedReward(state, action)

        return outcomes.map { (nextStateKey, count) ->
            @Suppress("UNCHECKED_CAST")
            ProbabilisticTransition(
                state = state,
                action = action,
                reward = avgReward,
                nextState = when(nextStateKey) {
                    is ComparableIntList -> mk.ndarray(nextStateKey.data).asDNArray()
                    else -> nextStateKey
                } as State,
                probability = count.toDouble() / total,
                done = false // Optional: could allow tracking terminal transitions
            )
        }
    }

    /**
     * Calculates the expected reward for a given state and action based on observed rewards and visit counts.
     * The expected reward is determined as the sum of rewards divided by the number of visits for the
     * specified state-action pair. If there are no recorded visits, default values are used.
     *
     * @param state The state for which the expected reward is being calculated.
     * @param action The action for which the expected reward is being calculated.
     * @return The expected reward for the given state-action pair. Returns 0.0 if there are no recorded rewards,
     * or 1.0 as a divisor if there are no recorded visit counts.
     */
    override fun expectedReward(state: State, action: Action): Double {
        val key = stateActionKey(state, action)
        val sum = rewardSums[key] ?: return 0.0
        val count = visitCounts[key] ?: return 1.0
        return sum / count
    }

    /**
     * Updates the internal representation of the model with a given state-action transition.
     * This method processes the transition to update state sets, terminal states,
     * transition counts, reward sums, visit counts, and predecessor mappings.
     *
     * @param transition The transition data containing the current state, action, reward,
     *                   next state, and a flag indicating whether the transition results
     *                   in a terminal state.
     */
    override fun update(transition: Transition<State, Action>) {
        val (state, action, reward, nextState) = transition
        val stateActionKey = stateActionKey(state, action)
        val stateKey = stateKey(state)
        val nextStateKey = stateKey(nextState)
        stateSet.add(stateKey)
        stateSet.add(nextStateKey)
        if (transition.done) terminalSet.add(nextStateKey)

        // Transition counts
        val nextMap = transitionCounts.getOrPut(stateActionKey) { mutableMapOf() }
        nextMap[nextStateKey] = nextMap.getOrDefault(nextStateKey, 0) + 1

        // Reward sums
        rewardSums[stateActionKey] = rewardSums.getOrDefault(stateActionKey, 0.0) + reward

        // Visit counts
        visitCounts[stateActionKey] = visitCounts.getOrDefault(stateActionKey, 0) + 1

        // Predecessor tracking
        predecessorsMap.getOrPut(nextStateKey) { mutableSetOf() }.add(stateActionKey)
    }

    /**
     * Samples a transition from the model based on observed state-action pairs and their corresponding outcomes.
     * The method selects a state-action key at random from the visitation counts, retrieves its associated outcomes,
     * and calculates an expected reward for the transition. If successful, it constructs and returns a transition
     * object containing the state, action, reward, next state, and termination flags. If no valid transition can be
     * sampled, the method returns null.
     *
     * @return A transition containing the state, action, reward, resulting state, and termination status if a valid
     * transition is found. Returns null if no valid transition can be sampled.
     */
    @Suppress("UNCHECKED_CAST")
    override fun sampleTransition(): Transition<State, Action>? {
        val keys = visitCounts.keys.shuffled()
        for (key in keys) {
            val state = when(key.state) {
                is ComparableIntList -> mk.ndarray(key.state.data).asDNArray()
                else -> key.state
            } as State
            val action = key.action as Action
            val outcomes = transitionCounts[key] ?: continue
            val nextState = outcomes.entries.randomOrNull()?.key ?: continue
            val avgReward = expectedReward(state, action)
            return Transition(
                state = state,
                action = action,
                reward = avgReward,
                nextState = when(nextState) {
                    is ComparableIntList -> mk.ndarray(nextState.data).asDNArray()
                    else -> nextState
                } as State,
                terminated = false,
                truncated = false
            )
        }
        return null
    }

    /**
     * Determines whether a given state-action pair is known in the model.
     *
     * @param state The state to check in the model.
     * @param action The action to check for the given state in the model.
     * @return True if the state-action pair has been recorded in the model, false otherwise.
     */
    override fun isKnown(state: State, action: Action): Boolean {
        val key = stateActionKey(state, action)
        return visitCounts.containsKey(key)
    }

    /**
     * Retrieves the set of predecessors for a given state. Predecessors are represented as
     * state-action keys that lead to the specified state in the Markov Decision Process model.
     *
     * @param state The state for which the predecessors are to be retrieved.
     * @return A set of state-action keys representing the predecessors of the given state.
     *         Returns an empty set if no predecessors are found for the given state.
     */
    override fun predecessors(state: State): Set<StateActionKey<*, *>> =
        predecessorsMap[stateKey(state)] ?: emptySet()

    /**
     * Retrieves the visit count for a specific state-action pair.
     * This value represents how many times the given state-action pair
     * has been observed or utilized within the model.
     *
     * @param state The state for which the visit count is being queried.
     * @param action The action associated with the state for the visit count query.
     * @return The number of visits recorded for the given state-action pair.
     * If the pair is not found, returns 0 by default.
     */
    override fun visitCount(state: State, action: Action): Int {
        return visitCounts.getOrDefault(stateActionKey(state, action), 0)
    }

    /**
     * Determines whether the given state is a terminal state in the model.
     * A terminal state is typically one that has no further transitions.
     *
     * @param state The state to be checked for terminality.
     * @return True if the given state is a terminal state, false otherwise.
     */
    override fun isTerminal(state: State): Boolean = terminalSet.contains(stateKey(state))
}
