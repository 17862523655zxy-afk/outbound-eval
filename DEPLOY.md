# 外呼 Agent 评测平台 - 部署指南

## 方式一：Railway 部署（推荐，免费）

### 步骤 1：准备 GitHub 仓库

```bash
# 在项目目录执行
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/你的用户名/outbound-eval.git
git push -u origin main
```

### 步骤 2：部署到 Railway

1. 访问 https://railway.app
2. 点击 "Start a New Project" → "Deploy from GitHub repo"
3. 选择你的仓库
4. Railway 会自动检测并部署

### 步骤 3：配置环境变量

在 Railway 项目设置中添加环境变量：

| 变量名 | 值 |
|--------|-----|
| `OPENAI_API_KEY` | 你的 API Key |
| `BASE_URL` | `https://api.deepseek.com` |
| `LLM_MODEL` | `deepseek-chat` |

### 步骤 4：获取访问地址

Railway 会自动分配一个域名，如：
`https://outbound-eval-production.up.railway.app`

---

## 方式二：Render 部署（免费）

1. 访问 https://render.com
2. 连接 GitHub 仓库
3. 选择 "Web Service"
4. 配置：
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python -m outbound_eval serve --host 0.0.0.0 --port $PORT`
5. 添加环境变量同上

---

## 方式三：Docker 部署

```bash
# 构建镜像
docker build -t outbound-eval .

# 运行容器
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e BASE_URL=https://api.deepseek.com \
  outbound-eval
```

---

## 必需的环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API Key | `sk-xxx` |
| `BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |

---

## 部署后验证

1. 访问分配的域名
2. 检查 `/api/v1/tasks` 接口是否返回任务列表
3. 尝试运行一次评测
