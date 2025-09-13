#!/usr/bin/env kotlin -Djava.awt.headless=false -Dprism.order=sw

@file:Repository("https://repo1.maven.org/maven2")
@file:Repository("https://central.sonatype.com/repository/maven-snapshots/")

@file:DependsOn("io.github.kotlinrl:integration:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:tabular:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:envs:0.1.0-SNAPSHOT")
@file:DependsOn("io.github.kotlinrl:rendering:0.1.0-SNAPSHOT")
@file:DependsOn("org.openjfx:javafx-base:17.0.10")
@file:DependsOn("org.openjfx:javafx-graphics:17.0.10")
@file:DependsOn("org.openjfx:javafx-controls:17.0.10")
@file:DependsOn("org.openjfx:javafx-media:17.0.10")

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.wrapper.saveBufferedImageAsPng
import io.github.kotlinrl.envs.*
import io.github.kotlinrl.rendering.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.kandy.letsplot.export.*
import java.io.*


val testEpisodes = 3

val env = Maze(render = true)
val recordEnv = RecordVideo(env, folder = "./videos/maze_policy_iteration")

val planner = policyIteration()
val (policy, vTable) = planner.plan(env.asMDP(0.9))

val trainer = episodicTrainer(
    agent = policyAgent(policy = policy.pi()),
    env = recordEnv,
    successfulTermination = { it.reward == 0.0 },
    closeOnSuccess = true,
)
val testResults = trainer.train(maxEpisodes(testEpisodes))
println("Test average reward: ${testResults.totalAverageReward}")
repeat(testEpisodes) {
    displayVideo(
        episode = it + 1,
        folder = recordEnv.folder
    )
}
printPolicyGrid(policy, 5, mapOf(0 to "↑", 1 to "→", 2 to "↓", 3 to "←"))
val plot = plotPolicyStateValueGrid(policy, vTable, 5, mapOf(0 to "↑", 1 to "→", 2 to "↓", 3 to "←"))
saveBufferedImageAsPng(plot.toBufferedImage(), File(recordEnv.folder, "policy_grid.png"))
