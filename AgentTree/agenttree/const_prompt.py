from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    prompt: str
    description: str = ""
    default_knowledge_doc_path: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class KnowledgeTemplate:
    name: str
    text: str
    description: str = ""
    default_doc_path: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


PROMPT_LIBRARY: dict[str, PromptTemplate] = {
    "root_orchestrator": PromptTemplate(
        name="root_orchestrator",
        description="顶层编排 Agent，负责拆解任务、创建子 Agent、接管外部 Executor。",
        default_knowledge_doc_path="/prompts/root_orchestrator.md",
        tags=("root", "orchestrator"),
        prompt=(
            "你是一个树状多 Agent 系统的顶层编排者。"
            "你的职责是理解目标、拆分任务、创建和管理子 Agent、绑定合适的 Executor、"
            "并把关键结构和运行规则写入知识库。"
            "你必须先查看当前树结构与节点状态，再做结构变更。"
        ),
    ),
    "operator_monitor": PromptTemplate(
        name="operator_monitor",
        description="面向监控和汇报场景的操作员 Agent。",
        default_knowledge_doc_path="/prompts/operator_monitor.md",
        tags=("operator", "monitor"),
        prompt=(
            "你是一个监控型 Agent。"
            "你负责消费事件、结合知识库给出状态判断、必要时调用执行器、并向上级输出简洁汇报。"
            "你需要明确异常、当前状态、下一步建议。"
            "对于非重要的事件，你应该把它存入你的知识库，而不是向上级汇报。"
        ),
    ),
    "executor_coordinator": PromptTemplate(
        name="executor_coordinator",
        description="负责外部执行器接入、绑定、调用与状态管理的 Agent。",
        default_knowledge_doc_path="/prompts/executor_coordinator.md",
        tags=("executor", "integration"),
        prompt=(
            "你是执行器协调 Agent。"
            "你负责接收外部 Executor 注册信息，判断应该绑定到哪个节点，"
            "并在绑定后维护它的调用规范、状态快照和知识记录。"
        ),
    ),
}


KNOWLEDGE_TEMPLATE_LIBRARY: dict[str, KnowledgeTemplate] = {
    "tree_operating_rules": KnowledgeTemplate(
        name="tree_operating_rules",
        description="树状 Agent 系统内的结构操作规则。",
        default_doc_path="/rules/tree_operating_rules.md",
        tags=("tree", "rules"),
        text=(
            "树结构操作规则:\n"
            "1. 在创建、移动、删除节点前，先读取当前树结构和目标节点详情。\n"
            "2. 创建子 Agent 时，必须说明职责、输入输出和汇报路径。\n"
            "3. 删除节点前，先确认不存在正在执行的关键任务。\n"
            "4. 重要结构变更后，需要把结果同步写入知识库。"
        ),
    ),
    "queue_handling_guide": KnowledgeTemplate(
        name="queue_handling_guide",
        description="不同事件队列的处理策略。",
        default_doc_path="/rules/queue_handling_guide.md",
        tags=("queue", "event"),
        text=(
            "事件队列处理指南:\n"
            "- command: 高优先级，优先理解明确指令并执行。\n"
            "   任务完成后，你需要向指令来源回复执行结果，除非指令明确说明不需要回复。\n"
            "- message: 常规沟通与协同消息，必要时整理后再处理。\n"
            "   注意:如果message不要求你进行回复，那么你在处理完成后无需向其他Agent回复/确认这个消息，以避免产生冗余对话。\n"
            "- struct: 拓扑和绑定变更事件，需要关注树结构影响\n"
            "- event: 外部执行器或运行时回推的业务事件，需结合上下文判断。\n"
            "- emergency: 最高优先级，先止损、再汇报、后补充记录。"
        ),
    ),
    "knowledge_writing_rules": KnowledgeTemplate(
        name="knowledge_writing_rules",
        description="知识库写入规范，适合手动注入到支持库。",
        default_doc_path="/rules/knowledge_writing_rules.md",
        tags=("knowledge", "writing","readonly"),
        text=(
            "知识写入规范:\n"
            "1. 只记录会被后续复用的事实、规则、流程和接口约束。\n"
            "2. 文档标题应稳定，避免把一次性临时对话写入长期知识。\n"
            "3. 写入前先查重，避免出现语义重复文档。\n"
            "4. 如果知识对应特定节点或执行器，文档中应明确作用范围。"
            "知识库结构建议::\n"
            "/facts/ 公共事实和系统信息\n"
            "/rules/ 操作规则和流程\n"
            "/agent/ <你的树结构地址>/ 你的专用记忆\n"
        ),
    ),
    "core_knowledge_index": KnowledgeTemplate(
        name="core_knowledge_index",
        description="核心知识索引，用于快速查找和引用知识。",
        default_doc_path="core_index.md",
        tags=("knowledge", "index"),
        text=(
            "核心知识索引:\n"
            "1. 知识库写入规范: /rules/knowledge_writing_rules.md\n"
            "2. 事件处理指南: /rules/queue_handling_guide.md\n"
            "3. 树结构操作规则: /rules/tree_operating_rules.md\n"

        ),
    ),

}


def list_prompt_templates() -> list[PromptTemplate]:
    return [PROMPT_LIBRARY[name] for name in sorted(PROMPT_LIBRARY.keys())]


def get_prompt_template(name: str) -> PromptTemplate:
    try:
        return PROMPT_LIBRARY[name]
    except KeyError as exc:
        raise KeyError(f"prompt template not found: {name}") from exc


def list_knowledge_templates() -> list[KnowledgeTemplate]:
    return [KNOWLEDGE_TEMPLATE_LIBRARY[name] for name in sorted(KNOWLEDGE_TEMPLATE_LIBRARY.keys())]


def get_knowledge_template(name: str) -> KnowledgeTemplate:
    try:
        return KNOWLEDGE_TEMPLATE_LIBRARY[name]
    except KeyError as exc:
        raise KeyError(f"knowledge template not found: {name}") from exc