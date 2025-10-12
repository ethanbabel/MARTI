from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import json

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
        self.global_steps.append(step)

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

# ---- 测试用例 ----

def test_chain_workflow():
    wf = WorkflowTrajectory("chain", "chain test", "chain_case", time.time())
    # 多 agent 多轮链式
    wf.record_step("A", 1, "start", "a1")
    wf.record_step("B", 1, "a1", "b1")
    wf.record_step("C", 1, "b1", "c1")
    wf.record_step("A", 2, "c1", "a2")
    wf.finalize("a2", 1.0)
    print("Chain Workflow:", json.dumps(wf.to_dict(), ensure_ascii=False, indent=2))


def test_debate_workflow():
    wf = WorkflowTrajectory("debate", "debate test", "mad_case", time.time())
    # 多 agent 辩论（MAD） 并行相同输入
    inputs = "topic"
    wf.record_step("A", 1, inputs, "a_resp", metadata={"batch_id": "debate"})
    wf.record_step("B", 1, inputs, "b_resp", metadata={"batch_id": "debate"})
    wf.record_step("C", 1, inputs, "c_resp", metadata={"batch_id": "debate"})
    wf.finalize("votes", 0.8)
    print("Debate Workflow:", json.dumps(wf.to_dict(), ensure_ascii=False, indent=2))


def test_aggregation_workflow():
    wf = WorkflowTrajectory("aggregate", "aggregate test", "moa_case", time.time())
    # 并行推理 + 聚合 (MoA)
    wf.record_step("X", 1, "data", "x1")
    wf.record_step("Y", 1, "data", "y1")
    wf.record_step("Z", 1, "data", "z1")
    # 聚合
    parents = [1, 2, 3]
    wf.record_step("M", 1, "x1|y1|z1", "m_agg", metadata={"parent_steps": parents, "role": "aggregator"})
    wf.finalize("m_agg", 0.9)
    print("Aggregation Workflow:", json.dumps(wf.to_dict(), ensure_ascii=False, indent=2))


def test_adaptive_workflow():
    wf = WorkflowTrajectory("adaptive", "adaptive test", "adaptive_case", time.time())
    # 自适应流程：根据上一输出选择下一 agent
    out1 = wf.record_step("Agent1", 1, "input", "optA").output_text
    # 如果输出是 optA 则由 AgentA 处理，否则 AgentB
    next_agent = "AgentA" if out1 == "optA" else "AgentB"
    wf.record_step(next_agent, 1, out1, "resolved")
    wf.finalize("resolved", 1.0)
    print("Adaptive Workflow:", json.dumps(wf.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_chain_workflow()
    test_debate_workflow()
    test_aggregation_workflow()
    test_adaptive_workflow()
