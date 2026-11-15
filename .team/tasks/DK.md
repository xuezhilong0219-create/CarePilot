# 任务 DK：检测本机 Docker，并验证 CarePilot 的容器化配置

负责人：DeepSeek Harness（dsh）。运行环境限制同之前：**不要 git commit**，不要用 uv。这个任务**不写代码、不改仓库里任何已有文件**，只做检测和验证，把事实如实报告。

## 背景

`Dockerfile`、`docker-compose.yml`、`.dockerignore` 是之前写的，**从来没有真正构建过**（只做过 YAML 语法检查）。用户想知道：本机有没有 Docker，配置到底能不能用。

## 硬性规则（先读）

1. **绝对不要读取或打印** `D:/ORCApj/CarePilot/.env`，也不要读取任何 `.env` 之外的密钥。这个 worktree 里没有 `.env`，下面第 3 步让你创建一个**假的**。
2. **不要安装 Docker、不要改系统设置、不要提权、不要尝试绕过沙箱。**如果沙箱不允许你执行 docker，就如实报告，到此为止。
3. **不要执行** `docker system prune`、`docker volume prune`、`docker image prune`、`docker rm`/`docker rmi` 指向你自己创建的对象以外的任何东西。用户机器上可能有别的容器和镜像，不要碰。你创建的对象名字必须以 `carepilot-verify` 开头，清理时只清理这些。
4. 不要凭推测写结论。每一条结论后面要附上你实际执行的命令和输出（前 5 行即可，错误信息要完整）。

## 第 1 步：检测（只读）

逐条执行并记录**原样输出**（失败也要记录完整的错误信息）：

- `docker --version`
- `docker compose version`（以及旧版 `docker-compose --version`）
- PowerShell：`Get-Command docker -ErrorAction SilentlyContinue | Format-List Source,Version`；cmd：`where docker`
- 检查这些路径是否存在（只检查是否存在，不要读里面的内容）：
  `C:\Program Files\Docker\Docker\Docker Desktop.exe`、`C:\Program Files\Docker\Docker\resources\bin\docker.exe`
- `docker info`（看守护进程是否在运行；输出很长，只记录前 15 行和任何 `ERROR`/`error` 行）
- `docker context ls`
- `wsl -l -v`（只列出发行版，不要在里面执行任何命令）
- `podman --version`（有就记录，没有就记录"不存在"）

## 第 2 步：给出分类结论（必须是下面四种之一）

- **A. 未安装**：`docker` 命令不存在，且上面的安装路径也不存在。
- **B. 已安装但守护进程没运行**：`docker --version` 成功，但 `docker info` 报连不上守护进程。
- **C. 沙箱执行不了**：命令因为权限、沙箱限制、管道访问被拒等原因失败，**无法判断**是否安装。这种情况**不能写成 A**。
- **D. 可用**：`docker info` 成功。

如果是 A、B、C：写完结论就**停止**，不要继续第 3 步，直接输出 REPORT。B 和 C 要给出用户可以自己执行的下一步命令（比如"打开 Docker Desktop，等右下角鲸鱼图标变绿后再试"）。

## 第 3 步：验证配置（只有 D 才做）

在你当前所在的 worktree 目录（即构建上下文）里：

1. 创建**假的** `.env`（这个文件已在 .gitignore 里，不会入库）：
   ```
   LLM_API_KEY=dummy-not-a-real-key
   LLM_BASE_URL=http://localhost:9/v1
   ```
2. `docker compose config`：记录是否通过，以及展开后的 `environment` 和 `volumes`。
3. `docker build -t carepilot-verify:latest .`：记录是否成功、耗时、最终镜像大小（`docker image ls carepilot-verify`）。构建会拉取 `python:3.12-slim` 和 `ghcr.io/astral-sh/uv:0.12.13`，需要联网，可能要几分钟；如果拉取失败，原样记录错误（可能是网络或镜像源问题，不是配置问题）。
4. **用 compose 启动**（README 里写的用法）：`docker compose -p carepilot-verify up -d`，把宿主机端口改到 `18000` 避免冲突（用环境变量或 `-f` 覆盖文件都行，**不要修改仓库里的 docker-compose.yml**）。等最多 90 秒让它变 healthy，期间每 10 秒记录一次 `docker compose -p carepilot-verify ps` 的状态。
5. 请求 `http://localhost:18000/health`，记录返回。再请求 `POST /tickets`（不带 `X-User-Id` 头），应该是 401；这能证明治理模式的身份检查在容器里生效。
6. 检查容器内运行身份：`docker compose -p carepilot-verify exec api id`，**应该是非 root**（用户名 `app`）。
7. **关键检查——数据目录能不能写**：容器以非 root 运行，而 compose 把宿主机 `./data` 挂到了 `/app/data`。治理模式要在 `/app/data` 里创建 `tickets.sqlite`、`gateway.sqlite`、`checkpoints.sqlite`，还要写 `data/traces/`。
   - 用 `docker compose -p carepilot-verify exec api ls -la /app/data` 看目录属主；
   - 带一个有效身份头创建工单：`curl -X POST localhost:18000/tickets -H "X-User-Id: c1" -H "content-type: application/json" -d '{"message":"hello"}'`（**注意：这会真的调用模型，但 LLM_BASE_URL 是假的，所以预期会失败**，我们要看的是它**怎么失败**——是"连不上模型"这种预期的错误，还是"Permission denied: /app/data/..."这种配置错误。把响应和 `docker compose logs api --tail 40` 的输出原样记录）。
8. **不挂载卷的场景**：`docker run --rm -p 18001:8000 --env-file .env --name carepilot-verify-plain carepilot-verify:latest`（后台运行，等 30 秒），请求 `/health` 和同样的 `POST /tickets`，记录是否出现 `Permission denied` / `unable to open database file`。这是单独运行镜像、不用 compose 的情况。
9. 如果你发现镜像里的 healthcheck 不工作（状态一直是 `starting` 或 `unhealthy`），记录 `docker inspect --format '{{json .State.Health}}' <容器>` 的输出。

## 第 4 步：清理（无论成功失败都要做）

只清理你创建的对象：
- `docker compose -p carepilot-verify down -v`
- `docker rm -f carepilot-verify-plain`（如果存在）
- `docker rmi carepilot-verify:latest`
- 删除你创建的假 `.env`
- 确认 `docker ps -a` 和 `docker image ls` 里没有 `carepilot-verify` 开头的残留；**不要动其他任何东西**。

## 输出：REPORT（控制在 60 行内）

```
## REPORT DK
结论分类: A | B | C | D（一句话说明依据）
检测命令与输出: （逐条，失败的写完整错误）
验证结果（只有 D）:
  compose config: 通过/失败 + 原因
  build: 成功/失败 + 耗时 + 镜像大小 + 失败的完整错误
  compose 启动: 是否 healthy，用了多久
  /health: 返回
  无身份头 POST /tickets: 状态码
  运行身份: id 的输出
  数据目录: /app/data 的属主；写入是否成功
  POST /tickets（假的模型地址）: 响应 + 日志里的关键行（区分"模型连不上"和"权限错误"）
  不挂卷运行: 同上
发现的问题（只列事实，不要修）:
没有验证的部分:
清理: 做了什么，确认无残留
```
