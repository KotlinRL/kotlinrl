### Basic Usage in Python: dynamically typed data
From the [Gymnasium documentation](https://gymnasium.farama.org/introduction/basic_usage/)
```python
import gymnasium

env = gymnasium.make("CartPole-v1", seed=123, render_mode="rgb_array")
state, _ = env.reset()
print(f"Starting state: {state}")

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated
    print(f"State: {state}, Reward: {reward}")

print(f"Episode finished! Total reward: {total_reward}")
env.close()
```

### Basic Usage in Kotlin: statically typed data
```kotlin
val env = gymnasium.make<CartPoleEnv>(CartPole_v1, seed=123, render=true)
val (state, _) = env.reset()
println("Starting state: $observation")

var episodeOver = false
var totalReward = 0.0f

while (!episodeOver) {
    val action = env.actionSpace.sample()
    var (state, reward, terminated, truncated, _) = env.step(action)
    totalReward += reward
    episodeOver = terminated || truncated
    println("State: $observation, Reward: $reward")
}
println("Episode finished!: Total reward: $totalReward")
env.close()
```
### Running these examples requires open-rl-gymnasium-grpc-server to be running

```
docker pull kotlinrl/open-rl-gymnasium-grpc-server:latest
docker run --rm -p 50051:50051 kotlinrl/open-rl-gymnasium-grpc-server:latest
```