package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.agent.PolicyAgent
import io.mockk.*
import kotlin.test.Test
import kotlin.test.assertEquals

class PolicyAgentTest {

    @Test
    fun `test act method calls policy with correct state and returns action`() {
        // Arrange
        val mockPolicy: Policy<String, Int> = mockk()
        val policyAgent = PolicyAgent(
            id = "test-agent",
            policy = mockPolicy
        )
        val testState = "state1"
        val testAction = 42

        every { mockPolicy.invoke(testState) } returns testAction

        // Act
        val result = policyAgent.act(testState)

        // Assert
        assertEquals(testAction, result)
        verify(exactly = 1) { mockPolicy.invoke(testState) }
    }

    @Test
    fun `test observe(transition) calls onTransition with correct transition`() {
        // Arrange
        val mockOnTransition: TransitionObserver<String, Int> = mockk(relaxed = true)
        val policyAgent = PolicyAgent(
            id = "test-agent",
            policy = mockk(relaxed = true),
            onTransition = mockOnTransition
        )
        val testTransition = Transition("state1", 42, 5.0, "state2", false, false)

        // Act
        policyAgent.observe(testTransition)

        // Assert
        verify(exactly = 1) { mockOnTransition.invoke(testTransition) }
    }

    @Test
    fun `test observe(trajectory, episode) calls onTrajectory with correct trajectory and episode`() {
        // Arrange
        val mockOnTrajectory: TrajectoryObserver<String, Int> = mockk(relaxed = true)
        val policyAgent = PolicyAgent(
            id = "test-agent",
            policy = mockk(relaxed = true),
            onTrajectory = mockOnTrajectory
        )
        val testTrajectory = listOf(
            Transition("state1", 42, 5.0, "state2", false, false),
            Transition("state2", 99, 10.0, "state3", true, false)
        )
        val testEpisode = 10

        // Act
        policyAgent.observe(testTrajectory, testEpisode)

        // Assert
        verify(exactly = 1) { mockOnTrajectory.invoke(testTrajectory, testEpisode) }
    }
}
