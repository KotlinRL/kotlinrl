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

### Basic demonstrations shows ```proof of concept``` examples integrating against Gymnasium Environments
- [BasicUsage](basic/BasicUsage.ipynb)
- [BasicAcrobotExample](basic/BasicAcrobotExample.ipynb)
- [BasicAntExample](basic/BasicAntExample.ipynb)
- [BasicBipedalWalkerExample](basic/BasicBipedalWalkerExample.ipynb)
- [BasicBlackjackExample](basic/BasicBlackjackExample.ipynb)
- [BasicCartPoleExample](basic/BasicCartPoleExample.ipynb)
- [BasicCliffWalkingExample](basic/BasicCliffWalkingExample.ipynb)
- [BasicFrozenLakeExample](basic/BasicFrozenLakeExample.ipynb)
- [BasicHalfCheetahExample](basic/BasicHalfCheetahExample.ipynb)
- [BasicHopperExample](basic/BasicHopperExample.ipynb)
- [BasicHumanoidExample](basic/BasicHumanoidExample.ipynb)
- [BasicHumanoidStandupExample](basic/BasicHumanoidStandupExample.ipynb)
- [BasicInvertedDoublePendulumExample](basic/BasicInvertedDoublePendulumExample.ipynb)
- [BasicInvertedPendulumExample](basic/BasicInvertedPendulumExample.ipynb)
- [BasicLunarLanderExample](basic/BasicLunarLanderExample.ipynb)
- [BasicMountainCarContinuousExample](basic/BasicMountainCarContinuousExample.ipynb)
- [BasicMountainCarExample](basic/BasicMountainCarExample.ipynb)
- [BasicPendulumExample](basic/BasicPendulumExample.ipynb)
- [BasicPusherExample](basic/BasicPusherExample.ipynb)
- [BasicReacherExample](basic/BasicReacherExample.ipynb)
- [BasicSwimmerExample](basic/BasicSwimmerExample.ipynb)
- [BasicTaxiExample](basic/BasicTaxiExample.ipynb)
- [BasicWalker2dExample](basic/BasicWalker2dExample.ipynb)

### The Kotlin Notebooks in this package demonstrate classic RL techniques against Gymnasium Environments
Dynamic Programming

- [MazeValueIteration](dynamic_programming/MazeValueIteration.ipynb)
- [MazePolicyIteration](dynamic_programming/MazePolicyIteration.ipynb) 

Monte Carlo Control
- [MazeIncrementalMonteCarloControl](monte_carlo_control/MazeIncrementalMonteCarloControl.ipynb)
- [MazeOffPolicyMonteCarloControl](monte_carlo_control/MazeOffPolicyMonteCarloControl.ipynb)
- [MazeOnPolicyMonteCarloControl](monte_carlo_control/MazeOnPolicyMonteCarloControl.ipynb)

Temporal Difference Learning 
- Classic Control
  - [CliffWalkingExpectedSARSA](classic_control/CliffWalkingExpectedSARSA.ipynb)
  - [CliffWalkingQLearning](classic_control/CliffWalkingQLearning.ipynb)
  - [CliffWalkingSARSA](classic_control/CliffWalkingSARSA.ipynb)
  - [FrozenLakeExpectedSARSA](classic_control/FrozenLakeExpectedSARSA.ipynb)
  - [FrozenLakeQLearning](classic_control/FrozenLakeQLearning.ipynb)
  - [FrozenLakeSARSA](classic_control/FrozenLakeSARSA.ipynb)
  - [TaxiExpectedSARSA](classic_control/TaxiExpectedSARSA.ipynb)
  - [TaxiQLearning](classic_control/TaxiQLearning.ipynb)
  - [TaxiSARSA](classic_control/TaxiSARSA.ipynb)