# AgentTree : 树状多Agent合作架构

## 概述

### 使用树状结构管理LLM Agents:

每个Agent都是树中的**节点**

- Agent之间**主要**沿着树结构传递信息，低一级Agent唤醒高级别Agent,高一级Agent**创建、管理和指导**下一级Agent,但**保留**跨级传递能力

- 基于向量的知识库也沿着节点分层

- Agent 可以自由设定用于自我触发的触发器

- Agent 可以自由创建自己的子节点,可以是Agent

- 工具(执行器)作为叶子节点接入树中，绑定到一个树中的Agent上

- 同时执行器不但可以被调用，同时也具有向所属Agent推送事件的能力，

- 各Agent通过工具事件队列被依次唤醒，并且处理时间

- Agent可以自由创建不同Agent数据通路

- 执行器不单单作为Agent的工具使用，而是也有自己独立运行/计算的能力
  
  比如，执行器可以是一段程序，由Agent调整它们的参数

- 当启动时，根节点Agent承担管理者的角色，按照它的Prompt规划结构，创建子Agent

- 人类可以和一个被称为supervisor的Agent交互，该Agent具有全部权限，并且没有任务队列，只负责接受人类的指令，并且进行相应操作或者向人类提供相应信息

---

## Quick Start

安装

```shell
cd AgentTree
pip install llm-agenttree[dev]
```

注意：不要在仓库根目录直接做 editable 安装；应始终进入 AgentTree 目录后再执行安装命令，否则可能残留错误的 agenttree editable 安装记录。

启动核心服务

```shell
agenttree-core
```

打开UI界面

```shell
agenttree-cli
```
