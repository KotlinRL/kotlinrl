package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*

class LookupModel<State, Action>(
    private val actions: List<Action>
) :LearnableMDPModel<State, Action>, ExpandableMDPModel<State, Action> {

    private val stateSet = mutableSetOf<Comparable<*>>()
    private val terminalSet = mutableSetOf<Comparable<*>>()
    private val transitionCounts = mutableMapOf<StateActionKey<*, *>, MutableMap<Comparable<*>, Int>>()
    private val rewardSums = mutableMapOf<StateActionKey<*, *>, Double>()
    private val visitCounts = mutableMapOf<StateActionKey<*, *>, Int>()
    private val predecessorsMap = mutableMapOf<Comparable<*>, MutableSet<StateActionKey<*, *>>>()

    override fun allStates(): List<State> = stateSet.map {
        @Suppress("UNCHECKED_CAST")
        when(it) {
            is ComparableIntList -> mk.ndarray(it.data).asDNArray()
            else -> it
        } as State
    }.toList()

    override fun allActions(): List<Action> = actions

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

    override fun expectedReward(state: State, action: Action): Double {
        val key = stateActionKey(state, action)
        val sum = rewardSums[key] ?: return 0.0
        val count = visitCounts[key] ?: return 1.0
        return sum / count
    }

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

    override fun isKnown(state: State, action: Action): Boolean {
        val key = stateActionKey(state, action)
        return visitCounts.containsKey(key)
    }

    override fun predecessors(state: State): Set<StateActionKey<*, *>> =
        predecessorsMap[stateKey(state)] ?: emptySet()

    override fun visitCount(state: State, action: Action): Int {
        return visitCounts.getOrDefault(stateActionKey(state, action), 0)
    }

    override fun isTerminal(state: State): Boolean = terminalSet.contains(stateKey(state))
}
