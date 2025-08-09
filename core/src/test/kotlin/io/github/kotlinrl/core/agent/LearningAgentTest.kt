package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.agent.LearningAgent
import io.mockk.*
import kotlin.test.Test
import kotlin.test.assertEquals

class LearningAgentTest {

    @Test
    fun `test act method delegates to algorithm`() {
        // Arrange
        val mockAlgorithm: LearningAlgorithm<String, Int> = mockk()
        val learningAgent = LearningAgent(
            id = "agent1",
            algorithm = mockAlgorithm
        )
        val testState = "state1"
        val testAction = 42

        every { mockAlgorithm.invoke(testState) } returns testAction

        // Act
        val result = learningAgent.act(testState)

        // Assert
        assertEquals(testAction, result)
        verify(exactly = 1) { mockAlgorithm.invoke(testState) }
    }

    @Test
    fun `test observe(transition) updates algorithm`() {
        // Arrange
        val mockAlgorithm: LearningAlgorithm<String, Int> = mockk(relaxed = true)
        val learningAgent = LearningAgent(
            id = "agent1",
            algorithm = mockAlgorithm
        )
        val testTransition = Transition("state1", 1, 10.0, "state2", false, false)

        // Act
        learningAgent.observe(testTransition)

        // Assert
        verify(exactly = 1) { mockAlgorithm.update(testTransition) }
    }

    @Test
    fun `test observe(trajectory, episode) updates algorithm`() {
        // Arrange
        val mockAlgorithm: LearningAlgorithm<String, Int> = mockk(relaxed = true)
        val learningAgent = LearningAgent(
            id = "agent1",
            algorithm = mockAlgorithm
        )
        val testTrajectory = listOf(
            Transition("state1", 1, 10.0, "state2", false, false),
            Transition("state2", 2, 20.0, "state3", true, false)
        )
        val testEpisode = 10

        // Act
        learningAgent.observe(testTrajectory, testEpisode)

        // Assert
        verify(exactly = 1) { mockAlgorithm.update(testTrajectory, testEpisode) }
    }
}
