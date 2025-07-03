package examples

import io.github.kotlinrl.integration.gymnasium.*
import io.github.kotlinrl.integration.gymnasium.GymnasiumEnvs.*

/*
Python Gymnasium code example
    import gymnasium

    env = gymnasium.make("CartPole-v1", seed=123, render_mode="rgb_array")
    observation, info = env.reset()
    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
        print(f"Observation: {observation}, Reward: {reward}")

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()
 */
fun main() {
    val env = gymnasium.make<CartPoleEnv>(CartPole_v1, seed=123, render=true)
    val (observation, _) = env.reset()
    println("Starting observation: $observation")

    var episodeOver = false
    var totalReward = 0.0

    while (!episodeOver) {
        val action = env.actionSpace.sample()
        val (observation, reward, terminated, truncated, _) = env.step(action)
        totalReward += reward
        episodeOver = terminated || truncated
        println("Observation: $observation, Reward: $reward")
    }
    println("Episode finished!: Total reward: $totalReward")
    env.close()
}







