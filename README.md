# Q-MacroAgent
量子驱动的多Agent中观行业分析系统

## 项目简介

Q-MacroAgent 是一个基于 AI 的智能公司研究分析系统，能够自动收集、分析和生成详细的公司研究报告。

## 主要功能

- 🔍 **智能公司研究**：自动收集公司信息、财务数据、新闻动态
- 📊 **多维度分析**：行业分析、竞争对手分析、市场趋势
- 📄 **报告生成**：自动生成专业的 PDF 研究报告
- 🌐 **Web 界面**：直观的前端界面，支持实时交互
- 🤖 **多 Agent 协作**：使用 LangGraph 构建的智能 Agent 工作流

## 技术栈

### 后端
- **Python 3.9+**
- **FastAPI** - Web 框架
- **LangChain** - AI 应用框架
- **LangGraph** - Agent 工作流
- **DeepSeek API** - 主要 AI 模型
- **Tavily** - 网络搜索 API

### 前端
- **React + TypeScript**
- **Vite** - 构建工具
- **Tailwind CSS** - 样式框架

## 快速开始

### 环境要求
- Python 3.9+
- Node.js 16+
- npm 或 yarn

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/zhuangyaoming/Q-MacroAgent.git
cd Q-MacroAgent
```

2. **安装后端依赖**
```bash
pip install -r requirements.txt
```

3. **安装前端依赖**
```bash
cd ui
npm install
cd ..
```

4. **配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
```

5. **启动服务**

后端服务：
```bash
python application.py
```

前端服务：
```bash
cd ui
npm run dev
```

6. **访问应用**
- 前端界面：http://localhost:5174
- 后端 API：http://localhost:8000

## 使用说明

1. 在前端界面输入公司名称
2. 可选填写公司网址、行业、总部位置等信息
3. 点击"开始研究"按钮
4. 系统将自动进行多轮研究和分析
5. 生成完整的研究报告并支持 PDF 下载

## 项目结构

```
Q-MacroAgent/
├── backend/                 # 后端代码
│   ├── classes/            # 数据模型
│   ├── nodes/              # Agent 节点
│   ├── services/           # 服务层
│   └── utils/              # 工具函数
├── ui/                     # 前端代码
│   ├── src/
│   │   ├── components/     # React 组件
│   │   └── types/          # TypeScript 类型
├── static/                 # 静态资源
├── knowledge_base/         # 知识库
└── requirements.txt        # Python 依赖
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
