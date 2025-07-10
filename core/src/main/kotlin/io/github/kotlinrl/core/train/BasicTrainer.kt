package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.env.*

class BasicTrainer<State, Action>(
    private val env: Env<State, Action, *, *>,
    private val agent: Agent<State, Action>,
    private val maxStepsPerEpisode: Int = 10_000,
    private val callbacks: List<EpisodeCallback<State, Action>> = emptyList()
) : Trainer {

    override fun train(episodes: Int): TrainingResult {
        val episodeRewards = mutableListOf<Double>()

        repeat(episodes) { episode ->
            callbacks.forEach { it.onEpisodeStart(episode) }

            val transitions = mutableListOf<Transition<State>>()
            val actions = mutableListOf<Action>()

            var priorState = env.reset().observation
            var priorAction: Action? = null
            var totalReward = 0.0
            var steps = 0

            repeat(maxStepsPerEpisode) {
                val action = agent.act(priorState)
                val transition = env.step(action)
                totalReward += transition.reward

                agent.observe(
                    Experience(
                        transition = transition,
                        priorState = priorState,
                        priorAction = priorAction
                    )
                )

                transitions += transition
                actions += action
                steps++

                priorState = transition.observation
                priorAction = action

                if (transition.terminated || transition.truncated) return@repeat
            }

            val stats = EpisodeStats(
                episode = episode,
                totalReward = totalReward,
                steps = steps,
                transitions = transitions,
                actions = actions
            )

            callbacks.forEach { it.onEpisodeEnd(stats) }
            episodeRewards += totalReward
        }

        return TrainingResult(episodeRewards)
    }
}
