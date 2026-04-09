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

- Agent可以自由创建不同Agent数据通路,类似消息群聊,这个

- 执行器不单单作为Agent的工具使用，而是也有自己独立运行/计算的能力
  
  比如，执行器可以是一段程序，由Agent调整它们的参数

- 当启动时，根节点Agent承担管理者的角色，按照它的Prompt规划结构，创建子Agent

- 人类可以和一个被称为supervisor的Agent交互，该Agent具有全部权限，并且没有任务队列，只负责接受人类的指令，并且进行相应操作或者向人类提供相应信息

### 知识库:

- 知识库按照和Agent树一样的结构分层，类似文件系统，比如 `/reactor/turbine/INTRO.md`

- 知识库基于向量进行检索(RAG),也允许直接获取文件内容

- Agent可以修改，和添加 知识库中的内容

- 当Agent树结构发生变化时，知识库也应该自动进行调整

- 同时，Agent检索知识库应该由它自己通过工具进行，而不是自动进行，并且它可以自己指定检索范围，示例如下

| 检索范围                              | 解释            | 返回结果示例                                              |
| --------------------------------- | ------------- | --------------------------------------------------- |
| `/reactor/core/`                  | 查询一个“文件夹下”的内容 | `reactor/core/control.md` `reactor/core/control.md` |
| `./`(当前Agent为`/reactor/tubrine/`) | 使用相对路径查询      | `./reactor/tubrine.md`                              |

### Agent管理/Agent 节点

- Agent可以创建自己的子Agent,相当于自己的子节点,并且为它编写提示词

- Agent也可以管理自己的子Agent,包括创建，命令，删除，修改提示词等操作都可以通过它自己的工具进行

- 每个Agent都拥有多个事件队列，按照优先级排序如下
  
  | 名称        | 解释                      | 优先级   |
  | --------- | ----------------------- | ----- |
  | command   | 上级Agent下发的命令            | 1(最高) |
  | message   | 下级Agent以及Agent群聊中的消息    | 2     |
  | struct    | 当Agent树结构发生变化等时候通知Agent | 3     |
  | emergency | 执行器发送的紧急事件              | 4     |
  | event     | 执行器发送的普通事件              | 5(最低) |

- Agent可以给其他任意一个树上的Agent节点发送消息

### 执行器

- 执行器是Agent的“手”和感官

- 执行器可以向Agent推送事件

- Agent可以通过工具调用执行器的能力

- 执行器是一段程序，通过网络链接到主服务上

- 执行器的所有权可以被转移和更改，由同一级以及上一级Agent自行管理和决定
  
  - 比如:Agent` /reactor/tubrine/core`决定把执行器` /reactor/tubrine/a.executor`的所有权转移给`/reactor/tubrine/operators/tubrine_operator_a`

### Broker

- 你应该创建一套系统，管理Agent之间事件/消息/数据的流动

- 可以参考MQTT协议中broker的设计思路

## 技术实现&&功能要求

1. 使用Python实现，后端服务使用Langchain(1.0+)+FastAPI

2. 对于Agent，使用langchain的 create_agent 得到一个ReAct Agent

3. 如果你对Langchain不熟悉，可以git clone [liwusen/FaustBot-llm-vtuber: FaustBot LLM Vtuber/桌宠](https://github.com/liwusen/FaustBot-llm-vtuber) 参考它的backend主程序

4. 实现一个core主服务，它不负责Agent的管理，而是负责数据的流转管理和对Agent教程管理

5. 每一个Agent和执行器一个作为一个子进程的形式被Core启动，通过网络链接到后端主服务

6. 把代码放在AgentTree子目录下，按照一个Python Module的形式组织代码

7. 编写完善的Demo程序，对功能进行测试



----



## 你的实现步骤和要求

1. 阅读这些要求，仔细思考给出你的计划和方案

2. 要求我审批和修改

3. 进行编码
