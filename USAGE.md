# SwarmGPT Extension 使用指南

本文档介绍如何在本机安装、配置和运行 SwarmGPT 扩展版。

---

## 目录

- [环境要求](#环境要求)
- [快速安装（一键）](#快速安装一键)
- [配置 API Key](#配置-api-key)
- [启动应用](#启动应用)
- [Web 界面使用](#web-界面使用)
- [扩展功能详解](#扩展功能详解)
- [Python API 使用](#python-api-使用)
- [常见问题](#常见问题)

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11（仿真模式）或 Linux x64（完整部署） |
| Python | 3.11 |
| Conda | Miniconda 或 Anaconda |
| 内存 | >= 8 GB |
| API Key | OpenAI 或 Anthropic |

> **注意：** Windows 上运行为仿真模式，无法连接真实无人机。连接无人机需要 Linux + ROS Noetic。

---

## 快速安装（一键）

### 1. 克隆仓库

```bash
git clone https://github.com/hhuang999/swarmGPT.git
cd swarmGPT
```

### 2. 创建 Conda 环境

```bash
conda create -n swarmgpt python=3.11 -y
```

### 3. 安装依赖

```bash
conda activate swarmgpt
pip install -e .
```

> 安装过程约需 3-5 分钟，会自动安装 120+ 个依赖包。

### 4. 验证安装

```bash
python test_env.py
```

预期输出：

```
==================================================
SwarmGPT Environment Test
==================================================
Python: 3.11.x

[OK] Motion primitives: 22 registered
[OK] PrimitiveComposer
[OK] Provider factory
[OK] ImageFormationConverter, FlightBounds
[OK] VoiceController
[OK] ARBridge
[OK] Custom primitive modules
[OK] UI module

==================================================
SUCCESS: All modules imported correctly!
```

---

## 配置 API Key

### 方式一：编辑启动脚本（Windows 推荐）

打开 `start_windows.bat`，修改以下两行：

```bat
set OPENAI_API_KEY=sk-your-openai-key
set ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

### 方式二：环境变量（命令行）

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (可选)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 方式三：key.sh 文件

编辑 `key.sh`，填入你的 Key，然后在终端 source：

```bash
source key.sh
```

---

## 启动应用

### 方式一：双击启动脚本（Windows）

直接双击 `start_windows.bat`

### 方式二：命令行启动

```bash
conda activate swarmgpt
cd /path/to/swarmGPT
python swarm_gpt/launch.py
```

启动成功后，终端会显示：

```
Running on local URL:  http://127.0.0.1:7860
```

在浏览器中打开该地址即可使用。

### 启动参数

```bash
# 指定 LLM 模型
python swarm_gpt/launch.py --model_id="gpt-4o"

# 禁用运动原语
python swarm_gpt/launch.py --use_motion_primitives=False
```

---

## Web 界面使用

### 基本流程

1. **选择歌曲** — 从音乐库中选择一首歌
2. **生成编舞** — 系统自动分析节拍并生成无人机编队
3. **预览仿真** — 在仿真查看器中预览效果
4. **修改优化** — 通过文字提示修改编舞
5. **部署执行** — 满意后部署到无人机（需要 Linux + ROS）

### 扩展面板

#### 自定义原语（Custom Primitive）

1. 点击 **"Custom Primitive"** 展开面板
2. 输入原语名称，如 `spiral_up`
3. 输入中文描述，如 `螺旋上升然后散开`
4. 可选：输入 JSON 参数
5. 点击 **"Generate Primitive"** 生成

#### 图像/草图输入（Image/Sketch Input）

1. 点击 **"Image / Sketch Input"** 展开面板
2. **上传图片** 或使用 **画板手绘** 一个形状
3. 拖动滑块选择无人机数量 (3-20)
4. 点击 **"Detect Shape"** 检测形状
5. 系统自动生成 3D 编队预览

#### 语音控制（Voice Control）

1. 点击 **"Voice Control"** 展开面板
2. 点击麦克风按钮 **录音**（需要浏览器授权）
3. 系统自动转录并解析中文命令
4. 查看解析结果，点击 **"Apply Command"** 执行

支持的中文指令示例：

| 指令 | 效果 |
|------|------|
| `全部提高30` | 所有无人机升高 30cm |
| `右边降低` | 右侧无人机降低（默认 30cm） |
| `前面旋转45` | 前排无人机旋转 45° |
| `全部散开` | 所有无人机散开 |
| `中间聚拢` | 中心区域无人机聚拢 |
| `左边左移` | 左侧无人机向左移动 |

---

## 扩展功能详解

### 多模型提供商

支持 OpenAI 和 Anthropic 两种 LLM：

```python
from swarm_gpt.providers import get_provider

# OpenAI
provider = get_provider("openai", api_key="sk-...")

# Anthropic
provider = get_provider("anthropic", api_key="sk-ant-...")

# 调用
response = provider.generate("生成一个心形编队")
```

### 6 个新运动原语

| 原语 | 参数 | 效果 |
|------|------|------|
| `firework(n, height, dir)` | 无人机数、高度、方向 | 烟花绽放 |
| `pendulum(n, period)` | 无人机数、周期 | 钟摆摆动 |
| `scatter_gather(n, spread, ...)` | 无人机数、扩散度 | 散开聚拢 |
| `form_heart(n_repeats, height)` | 重复次数、高度 | 心形编队 |
| `form_line(n, axis)` | 无人机数、轴向 | 直线编队 |
| `orbit(center, radius, speed)` | 圆心、半径、速度 | 环绕飞行 |

### 可组合原语（YAML 语法）

在编舞 YAML 中使用组合运算符：

```yaml
# 顺序执行
op: sequence
children:
  - primitive: form_heart
    params: [1.0, 150]
  - primitive: orbit
    params: [[0, 0], 100, 20]

# 分组并行
op: parallel
drone_groups: [[0, 1, 2], [3, 4, 5]]
children:
  - primitive: firework
    params: [3, 150, 80]
  - primitive: pendulum
    params: [3, 2.0]

# 加权混合
op: blend
weight: 0.7
children:
  - primitive: rotate
    params: [30, 'z']
  - primitive: wave
    params: [3, 100, [[1,1],[2,2]], [0.5,0.3], [0.2,0.4]]
```

### 自定义原语生成器

```python
from swarm_gpt.core.backend import AppBackend

backend = AppBackend()
result = backend.create_custom_primitive(
    description="螺旋上升然后散开",
    name="spiral_scatter",
    params={"radius": "旋转半径 (cm)", "height": "上升高度 (cm)"}
)

print(result)  # {"success": True, "name": "spiral_scatter", ...}
```

### 图像转编队

```python
from swarm_gpt.core.multimodal.image_to_formation import ImageFormationConverter
import cv2

image = cv2.imread("heart_sketch.png")

converter = ImageFormationConverter()
positions, shape_name, metadata = converter.convert(
    image,
    n_drones=6,
    strategy="auto"   # "cv" | "vlm" | "auto"
)

print(positions.shape)   # (6, 3) — 单位 cm
print(shape_name)        # "cv_contour" 或 "vlm_shape"
print(metadata.strategy)  # 使用的策略
```

### 中文语音控制

```python
from swarm_gpt.core.multimodal.voice_controller import VoiceController
import numpy as np

vc = VoiceController()  # 文字解析模式
# vc = VoiceController(openai_client=client)  # 语音转文字模式

positions = np.array([
    [-90, -60, 100], [-30, -60, 100], [30, -60, 100],
    [-90,  60, 100], [-30,  60, 100], [30,  60, 100],
])

result = vc.parse_command("右边的提高30", positions)

print(result["action"])         # "提高"
print(result["target_drones"])  # [2, 5] — 右侧无人机索引
print(result["primitive_call"]) # "move_z([3, 6], 30)"
print(result["fallback"])       # False
```

### AR 预览桥接

```python
from swarm_gpt.core.multimodal.ar_bridge import ARBridge

bridge = ARBridge(host="0.0.0.0", port=8000)

# 设置编队
bridge.set_formation(positions, metadata={"shape": "heart"})

# 处理命令
bridge.process_command({
    "type": "move_drone",
    "id": 0,
    "position": [50, 50, 100]
})

# 启动服务（阻塞）
# bridge.start()
```

**API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `ws://host:8000/ws/positions` | WebSocket | 实时位置推送 |
| `http://host:8000/api/formation` | POST | 设置编队 |
| `http://host:8000/api/command` | POST | 发送命令 |
| `http://host:8000/api/health` | GET | 健康检查 |

---

## Python API 使用

### 完整示例：从图像到编舞

```python
import cv2
import numpy as np
from swarm_gpt.core.multimodal.image_to_formation import ImageFormationConverter
from swarm_gpt.core.multimodal.voice_controller import VoiceController
from swarm_gpt.core.motion_primitives import primitive_by_name

# 1. 从图像生成编队位置
image = cv2.imread("star_shape.png")
converter = ImageFormationConverter()
positions, shape_name, metadata = converter.convert(image, n_drones=6)

# 2. 语音微调
vc = VoiceController()
result = vc.parse_command("全部提高50", positions)

# 3. 执行运动原语
limits = {
    "lower": np.array([-2.0, -2.0, 0.0]),
    "upper": np.array([2.0, 2.0, 2.0]),
}
firework_fn = primitive_by_name("firework")
final_pos, waypoints = firework_fn((6, 150, 80), positions, 0.0, 4.0, limits)
```

### 完整示例：可组合编舞

```python
from swarm_gpt.core.primitive_composer import PrimitiveComposer
from swarm_gpt.core.motion_primitives import primitive_by_name
import numpy as np

composer = PrimitiveComposer()

# 解析组合表达式
tree = composer.parse_composition_yaml({
    "op": "sequence",
    "children": [
        {"primitive": "form_heart", "params": [1.0, 150]},
        {"primitive": "firework", "params": [6, 200, 80]},
    ]
})

# 执行
swarm_pos = np.array([...])  # (n_drones, 3) cm
limits = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}
final_pos, waypoints = composer.execute_composed(tree, swarm_pos, 0.0, 8.0, limits)
```

---

## 常见问题

### Q: Windows 上报错 `No module named 'rospy'`

这是正常的。Windows 不支持 ROS。本项目已在 `swarm_gpt/__init__.py` 中添加了自动 mock，如果仍然报错，请确保通过 `python swarm_gpt/launch.py` 或 `python test_env.py` 启动，而不是直接导入子模块。

### Q: `OPENAI_API_KEY` 相关错误

确保已设置 API Key：

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# Windows CMD
set OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

### Q: 测试不通过

部分测试依赖 ROS 环境，在 Windows 上需要忽略：

```bash
python -m pytest tests/ -q --ignore=tests/test_providers --ignore=tests/unit/test_backend.py
```

### Q: 如何切换 Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export SWARMGPT_PROVIDER="anthropic"
```

### Q: 如何更新代码

```bash
cd swarmGPT
git pull origin main
pip install -e .
```

### Q: 如何卸载环境

```bash
conda remove -n swarmgpt --all
```

---

## 项目结构

```
swarmGPT/
├── swarm_gpt/
│   ├── __init__.py                    # 入口（含 Windows mock）
│   ├── launch.py                      # 启动脚本
│   ├── core/
│   │   ├── motion_primitives.py       # 运动原语（22 个）
│   │   ├── primitive_composer.py      # 组合运算符
│   │   ├── custom_primitive_generator.py  # 自定义原语生成器
│   │   ├── backend.py                 # 应用后端
│   │   ├── choreographer.py           # 编舞引擎
│   │   └── multimodal/
│   │       ├── __init__.py
│   │       ├── image_to_formation.py  # 图像转编队
│   │       ├── voice_controller.py    # 中文语音控制
│   │       └── ar_bridge.py           # AR 预览桥接
│   ├── providers/
│   │   ├── __init__.py                # get_provider 工厂
│   │   ├── base.py                    # LLMProvider 抽象类
│   │   ├── openai_provider.py         # OpenAI 实现
│   │   └── anthropic_provider.py      # Anthropic 实现
│   └── ui/
│       └── ui.py                      # Gradio Web UI
├── tests/
│   ├── test_primitives/               # 原语测试
│   ├── test_multimodal/               # 多模态测试
│   ├── test_providers/                # 提供商测试
│   ├── test_ui/                       # UI 测试
│   └── test_integration/              # 集成测试
├── start_windows.bat                  # Windows 启动脚本
├── start.sh                           # Linux/Mac 启动脚本
├── key.sh                             # API Key 模板
├── test_env.py                        # 环境测试脚本
└── pyproject.toml                     # 项目配置
```

---

## 引用

```bibtex
@article{schuck2025swarmgpt,
  title={SwarmGPT: Combining Large Language Models with Safe Motion Planning for Drone Swarm Choreography},
  author={Schuck, Martin and Dahanaggamaarachchi, Dinushka Orrin and Sprenger, Ben and Vyas, Vedant and Zhou, Siqi and Schoellig, Angela P.},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
