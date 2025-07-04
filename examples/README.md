### Basic Usage in Python: dynamically typed data
From the [Gymnasium documentation](https://gymnasium.farama.org/introduction/basic_usage/)
```python
import gymnasium

env = gymnasium.make("CartPole-v1", seed=123, render_mode="rgb_array")
observation, _ = env.reset()
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
```

### Basic Usage in Kotlin: statically typed data
```kotlin
val env: CartPoleEnv = gymnasium.make(CartPole_v1, seed=123, render=true)
val (observation, _) = env.reset()
println("Starting observation: $observation")

var episodeOver = false
var totalReward = 0.0f

while (!episodeOver) {
    val action = env.actionSpace.sample()
    val (observation, reward, terminated, truncated, _) = env.step(action)
    totalReward += reward
    episodeOver = terminated || truncated
    println("Observation: $observation, Reward: $reward")
}
println("Episode finished!: Total reward: $totalReward")
env.close()
```
### Running these examples requires open-rl-gymnasium-grpc-server to be running

```
docker pull kotlinrl/open-rl-gymnasium-grpc-server:latest
docker run --rm -p 50051:50051 kotlinrl/open-rl-gymnasium-grpc-server:latest
```