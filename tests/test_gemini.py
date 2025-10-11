import unittest
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# ==============================================================================
#  在这里粘贴您提供的原始代码 (AgentStep, AgentTrajectory, WorkflowTrajectory)
# ==============================================================================

@dataclass
class AgentStep:
    agent_name: str
    turn_id: int
    global_turn_id: int
    timestamp: float
    input_text: str
    output_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    tools_used: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "turn_id": self.turn_id,
            "global_turn_id": self.global_turn_id,
            "timestamp": self.timestamp,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tools_used": self.tools_used,
            "metadata": self.metadata,
        }

@dataclass
class AgentTrajectory:
    agent_name: str
    steps: List[AgentStep] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_step(self, step: AgentStep):
        self.steps.append(step)
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "steps": [s.to_dict() for s in self.steps],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

@dataclass
class WorkflowTrajectory:
    workflow_id: str
    prompt: str
    label: str
    start_time: float
    end_time: Optional[float] = None
    agent_trajectories: Dict[str, AgentTrajectory] = field(default_factory=dict)
    global_metadata: Dict[str, Any] = field(default_factory=dict)
    final_output: Optional[str] = None
    final_reward: Optional[float] = None
    global_steps: List[AgentStep] = field(default_factory=list)
    _global_turn_counter: int = 0

    def get_or_create_agent_trajectory(self, agent_name: str) -> AgentTrajectory:
        if agent_name not in self.agent_trajectories:
            self.agent_trajectories[agent_name] = AgentTrajectory(agent_name)
        return self.agent_trajectories[agent_name]

    def add_agent_step(self, step: AgentStep):
        self.get_or_create_agent_trajectory(step.agent_name).add_step(step)
        # Sort global_steps by global_turn_id after adding
        self.global_steps.append(step)
        self.global_steps.sort(key=lambda s: s.global_turn_id)

    def record_step(
        self,
        agent_name: str,
        turn_id: int,
        input_text: str,
        output_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tools_used: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentStep:
        self._global_turn_counter += 1
        step = AgentStep(
            agent_name=agent_name,
            turn_id=turn_id,
            global_turn_id=self._global_turn_counter,
            timestamp=time.time(),
            input_text=input_text,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tools_used=tools_used or {},
            metadata=metadata or {},
        )
        self.add_agent_step(step)
        return step

    def finalize(self, final_output: str, final_reward: float):
        self.end_time = time.time()
        self.final_output = final_output
        self.final_reward = final_reward

    def get_duration(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "prompt": self.prompt,
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.get_duration(),
            "agent_trajectories": {name: traj.to_dict() for name, traj in self.agent_trajectories.items()},
            "global_metadata": self.global_metadata,
            "final_output": self.final_output,
            "final_reward": self.final_reward,
            "global_steps": [s.to_dict() for s in self.global_steps],
        }

# ==============================================================================
#  测试辅助函数
# ==============================================================================

def simulate_agent_run(agent_name: str, input_text: str) -> str:
    """模拟一个 Agent 的执行，返回一个固定的、可预测的输出。"""
    return f"Output from {agent_name} based on input: '{input_text[:30]}...'"

def print_trajectory_as_graph(trajectory: WorkflowTrajectory):
    """
    将 WorkflowTrajectory 可视化打印出来，模拟轨迹回溯。
    这实现了用户要求的 "能够按照轨迹重新遍历打印出来，比如结合图之类的"。
    """
    print("\n" + "="*90)
    print(f"🔄 Trajectory Playback for Workflow ID: {trajectory.workflow_id}")
    print(f"📝 Prompt: {trajectory.prompt}")
    print(f"⏱️ Duration: {trajectory.get_duration():.4f}s | Final Reward: {trajectory.final_reward}")
    print("-" * 90)
    print("Execution Flow (Chronological Order):\n")

    for step in trajectory.global_steps:
        # 简单的图形化表示
        prefix = f"║ [Step {step.global_turn_id}]"
        print(f"{prefix}─── Agent: {step.agent_name} (Agent's Turn #{step.turn_id})")
        print(f"║      ├─ Input:  \"{step.input_text[:80]}{'...' if len(step.input_text) > 80 else ''}\"")
        print(f"║      ├─ Output: \"{step.output_text[:80]}{'...' if len(step.output_text) > 80 else ''}\"")
        print(f"║      └─ Tokens: (In: {step.input_tokens}, Out: {step.output_tokens}) | Tools: {step.tools_used or 'None'}")
    
    print("\n" + "─" * 90)
    print(f"🏆 Final Output: {trajectory.final_output}")
    print("=" * 90 + "\n")


# ==============================================================================
#  测试用例类
# ==============================================================================

class TestWorkflowTrajectory(unittest.TestCase):

    def test_chain_workflow(self):
        """
        测试场景 1: 链式工作流 (Chain) ⛓️
        Planner -> Coder -> Executor
        """
        print("\n--- Running Test: ⛓️ Chain Workflow ---")
        workflow = WorkflowTrajectory(
            workflow_id="chain-workflow-001",
            prompt="Develop a feature to summarize text.",
            label="feature-development",
            start_time=time.time()
        )

        # 1. Planner Agent
        planner_input = workflow.prompt
        planner_output = simulate_agent_run("Planner", planner_input)
        workflow.record_step(
            agent_name="Planner", turn_id=1,
            input_text=planner_input, output_text=planner_output,
            input_tokens=15, output_tokens=50
        )

        # 2. Coder Agent
        coder_input = planner_output
        coder_output = simulate_agent_run("Coder", coder_input)
        workflow.record_step(
            agent_name="Coder", turn_id=1,
            input_text=coder_input, output_text=coder_output,
            input_tokens=50, output_tokens=150,
            tools_used={"code_linter": 1, "api_lookup": 3}
        )

        # 3. Executor Agent
        executor_input = coder_output
        executor_output = simulate_agent_run("Executor", executor_input)
        workflow.record_step(
            agent_name="Executor", turn_id=1,
            input_text=executor_input, output_text=executor_output,
            input_tokens=150, output_tokens=25
        )

        workflow.finalize(final_output=executor_output, final_reward=0.95)

        # --- Assertions ---
        self.assertEqual(workflow._global_turn_counter, 3)
        self.assertEqual(len(workflow.global_steps), 3)
        self.assertIn("Planner", workflow.agent_trajectories)
        self.assertIn("Coder", workflow.agent_trajectories)
        self.assertIn("Executor", workflow.agent_trajectories)
        self.assertEqual(len(workflow.agent_trajectories["Planner"].steps), 1)
        self.assertEqual(workflow.agent_trajectories["Coder"].total_input_tokens, 50)
        self.assertEqual(workflow.agent_trajectories["Coder"].total_output_tokens, 150)
        self.assertIsNotNone(workflow.end_time)
        self.assertEqual(workflow.final_output, executor_output)
        
        # --- Trajectory Playback ---
        print_trajectory_as_graph(workflow)

    def test_debate_workflow_mad(self):
        """
        测试场景 2: 辩论式工作流 (Multi-agent Debate) 🗣️
        Agent_Pro <=> Agent_Con (2 rounds), then Judge decides.
        """
        print("\n--- Running Test: 🗣️ Multi-agent Debate (MAD) Workflow ---")
        workflow = WorkflowTrajectory(
            workflow_id="debate-workflow-002",
            prompt="Should AI be regulated?",
            label="debate",
            start_time=time.time()
        )

        # Round 1
        pro_input_1 = workflow.prompt
        pro_output_1 = simulate_agent_run("Agent_Pro", pro_input_1)
        workflow.record_step("Agent_Pro", 1, pro_input_1, pro_output_1, 8, 80)
        
        con_input_1 = pro_output_1
        con_output_1 = simulate_agent_run("Agent_Con", con_input_1)
        workflow.record_step("Agent_Con", 1, con_input_1, con_output_1, 80, 95)

        # Round 2
        pro_input_2 = con_output_1
        pro_output_2 = simulate_agent_run("Agent_Pro", pro_input_2)
        workflow.record_step("Agent_Pro", 2, pro_input_2, pro_output_2, 95, 110)

        con_input_2 = pro_output_2
        con_output_2 = simulate_agent_run("Agent_Con", con_input_2)
        workflow.record_step("Agent_Con", 2, con_input_2, con_output_2, 110, 105)

        # Judge
        judge_input = f"Debate results:\nPRO: {pro_output_1}\n{pro_output_2}\nCON: {con_output_1}\n{con_output_2}"
        judge_output = simulate_agent_run("Judge", "Final verdict based on debate...")
        workflow.record_step("Judge", 1, judge_input, judge_output, 400, 50)

        workflow.finalize(judge_output, 1.0)

        # --- Assertions ---
        self.assertEqual(workflow._global_turn_counter, 5)
        self.assertEqual(len(workflow.agent_trajectories), 3) # Pro, Con, Judge
        self.assertEqual(len(workflow.agent_trajectories["Agent_Pro"].steps), 2)
        self.assertEqual(len(workflow.agent_trajectories["Agent_Con"].steps), 2)
        self.assertEqual(workflow.agent_trajectories["Agent_Pro"].total_input_tokens, 8 + 95)
        self.assertEqual(workflow.agent_trajectories["Agent_Con"].total_output_tokens, 95 + 105)
        self.assertEqual(workflow.global_steps[3].agent_name, "Agent_Con") # Check order
        self.assertEqual(workflow.global_steps[3].turn_id, 2)
        
        # --- Trajectory Playback ---
        print_trajectory_as_graph(workflow)

    def test_aggregation_workflow_moa(self):
        """
        测试场景 3: 聚合式工作流 (Mixture-of-Agents) 🧩
        User -> [Expert_A, Expert_B, Expert_C] -> Aggregator
        """
        print("\n--- Running Test: 🧩 Mixture-of-Agents (MoA) Workflow ---")
        workflow = WorkflowTrajectory(
            workflow_id="moa-workflow-003",
            prompt="What are the key benefits of serverless computing?",
            label="qa-aggregation",
            start_time=time.time()
        )

        # Parallel Experts
        expert_a_output = simulate_agent_run("Expert_A_Cost", workflow.prompt)
        workflow.record_step("Expert_A_Cost", 1, workflow.prompt, expert_a_output, 20, 120)

        expert_b_output = simulate_agent_run("Expert_B_Scale", workflow.prompt)
        workflow.record_step("Expert_B_Scale", 1, workflow.prompt, expert_b_output, 20, 130)

        expert_c_output = simulate_agent_run("Expert_C_Ops", workflow.prompt)
        workflow.record_step("Expert_C_Ops", 1, workflow.prompt, expert_c_output, 20, 110)

        # Aggregator
        aggregator_input = json.dumps({
            "cost_expert": expert_a_output,
            "scale_expert": expert_b_output,
            "ops_expert": expert_c_output
        })
        aggregator_output = simulate_agent_run("Aggregator", aggregator_input)
        workflow.record_step("Aggregator", 1, aggregator_input, aggregator_output, 360, 200)

        workflow.finalize(aggregator_output, 0.98)

        # --- Assertions ---
        self.assertEqual(workflow._global_turn_counter, 4)
        self.assertEqual(len(workflow.agent_trajectories), 4) # 3 Experts + 1 Aggregator
        self.assertEqual(len(workflow.agent_trajectories["Expert_A_Cost"].steps), 1)
        self.assertEqual(len(workflow.agent_trajectories["Aggregator"].steps), 1)
        # Check that the aggregator's input is correct
        self.assertIn("cost_expert", workflow.global_steps[3].input_text)
        self.assertIn("scale_expert", workflow.global_steps[3].input_text)
        self.assertEqual(workflow.agent_trajectories["Expert_B_Scale"].total_input_tokens, 20)
        
        # --- Trajectory Playback ---
        print_trajectory_as_graph(workflow)

    def test_adaptive_workflow(self):
        """
        测试场景 4: 自适应工作流 (Adaptive Routing) 🧭
        User -> Router -> [Coder | Writer]
        """
        print("\n--- Running Test: 🧭 Adaptive Routing Workflow ---")
        
        # --- Path 1: Routing to Coder ---
        prompt_code = "Write a python script for me"
        workflow_code = WorkflowTrajectory(
            workflow_id="adaptive-workflow-code-004",
            prompt=prompt_code,
            label="routing",
            start_time=time.time()
        )

        # Router decides
        router_input = prompt_code
        # Simulate router's logic: it outputs the name of the next agent
        router_output = "Coder" 
        workflow_code.record_step("Router", 1, router_input, router_output, 10, 2, metadata={"decision": "code_task"})

        # Call the selected agent
        self.assertEqual(router_output, "Coder")
        coder_input = prompt_code
        coder_output = simulate_agent_run("Coder", coder_input)
        workflow_code.record_step("Coder", 1, coder_input, coder_output, 10, 70)
        
        workflow_code.finalize(coder_output, 0.9)

        # --- Assertions for Path 1 ---
        self.assertEqual(workflow_code._global_turn_counter, 2)
        self.assertIn("Router", workflow_code.agent_trajectories)
        self.assertIn("Coder", workflow_code.agent_trajectories)
        self.assertNotIn("Writer", workflow_code.agent_trajectories)
        self.assertEqual(workflow_code.global_steps[0].metadata["decision"], "code_task")

        # --- Trajectory Playback for Path 1 ---
        print_trajectory_as_graph(workflow_code)

        # --- Path 2: Routing to Writer ---
        prompt_write = "Write a blog post about travel"
        workflow_write = WorkflowTrajectory(
            workflow_id="adaptive-workflow-write-005",
            prompt=prompt_write,
            label="routing",
            start_time=time.time()
        )

        # Router decides
        router_input = prompt_write
        router_output = "Writer" # Simulate different decision
        workflow_write.record_step("Router", 1, router_input, router_output, 10, 2, metadata={"decision": "writing_task"})

        # Call the selected agent
        self.assertEqual(router_output, "Writer")
        writer_input = prompt_write
        writer_output = simulate_agent_run("Writer", writer_input)
        workflow_write.record_step("Writer", 1, writer_input, writer_output, 10, 150)
        
        workflow_write.finalize(writer_output, 0.92)

        # --- Assertions for Path 2 ---
        self.assertEqual(workflow_write._global_turn_counter, 2)
        self.assertIn("Router", workflow_write.agent_trajectories)
        self.assertIn("Writer", workflow_write.agent_trajectories)
        self.assertNotIn("Coder", workflow_write.agent_trajectories)
        
        # --- Trajectory Playback for Path 2 ---
        print_trajectory_as_graph(workflow_write)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)