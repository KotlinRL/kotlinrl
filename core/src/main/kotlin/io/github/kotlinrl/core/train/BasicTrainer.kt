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

            val trajectories = mutableListOf<Trajectory<State, Action>>()
            val actions = mutableListOf<Action>()

            var state = env.reset().state
            var totalReward = 0.0
            var steps = 0
            var done = false

            while (!done && steps < maxStepsPerEpisode) {
                val action = agent.act(state)
                val transition = env.step(action)
                totalReward += transition.reward

                val trajectory = Trajectory(
                    state = state,
                    action = action,
                    nextState = transition.state,
                    reward = transition.reward,
                    terminated = transition.terminated,
                    truncated = transition.truncated,
                    info = transition.info
                )

                agent.observe(trajectory)

                trajectories += trajectory
                actions += action
                steps++

                state = transition.state
                done = transition.terminated || transition.truncated
            }

            val stats = EpisodeStats(
                episode = episode,
                totalReward = totalReward,
                steps = steps,
                trajectories = trajectories
            )

            callbacks.forEach { it.onEpisodeEnd(stats) }
            episodeRewards += totalReward
        }

        return TrainingResult(episodeRewards)
    }
}
