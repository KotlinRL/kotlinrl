package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

class EmpiricalMDPModel<State, Action>(
    private val env: ModelBasedEnv<State, Action, *, *>,
    private val allStates: List<State> = emptyList(),
    private val allActions: List<Action> = emptyList(),
    private val numSamples: Int = 100
) : MDPModel<State, Action> {

    override fun transitions(state: State, action: Action): ProbabilisticTrajectory<State, Action> {
        val outcomes = mutableMapOf<Transition<State, Action>, Int>()

        repeat(numSamples) {
            val (nextState, reward, terminated, truncated) = env.simulateStep(state, action)
            val key = Transition(state, action, reward, nextState, terminated, truncated)
            outcomes[key] = outcomes.getOrDefault(key, 0) + 1
        }

        return outcomes.entries.map { (key, count) ->
            val transition = key
            ProbabilisticTransition(
                state = transition.state,
                action = transition.action,
                reward = transition.reward,
                nextState = transition.nextState,
                probability = count.toDouble() / numSamples,
                done = transition.done
            )
        }
    }

    override fun expectedReward(state: State, action: Action): Double {
        var totalReward = 0.0

        repeat(numSamples) {
            val (_, reward, _, _) = env.simulateStep(state, action)
            totalReward += reward
        }

        return totalReward / numSamples
    }

    override fun allStates(): List<State> = allStates

    override fun allActions(): List<Action> = allActions
}
