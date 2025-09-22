# 中国大陆地区环境下的项目配置指南

本文档提供了在中国大陆地区使用国内网络环境进行项目配置的方法，适用于 Docker 和非 Docker 用户。

## 非 Docker 用户配置

非 Docker 用户可以直接使用以下命令进行配置：

```bash
make setup-cn
```

## Docker 用户配置

Docker 用户需要修改 Docker 的配置文件，以使用国内镜像源。以下是不同操作系统的配置方法：

### Mac 系统

#### 方法一：通过 Docker Desktop 界面配置

1. 点击桌面顶部菜单栏的 Docker 图标
2. 选择 "Settings" 或 "首选项"
3. 点击 "Docker Engine" 选项卡
4. 将以下配置复制到配置编辑器中（注意保留其他已有配置）：

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
```

5. 点击 "Apply & Restart" 按钮应用更改并重启 Docker

#### 方法二：直接修改配置文件

1. 打开终端
2. 使用文本编辑器打开 Docker 配置文件：

```bash
mkdir -p ~/.docker
nano ~/.docker/daemon.json
```

3. 将以下内容复制到文件中：

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
```

4. 保存文件（在 nano 中按 Ctrl+O，然后按 Enter，再按 Ctrl+X 退出）
5. 重启 Docker Desktop

### Linux 系统

1. 创建或编辑 Docker 配置文件：

```bash
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
```

2. 将以下内容复制到文件中：

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
```

3. 保存文件并重启 Docker 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Windows 系统

1. 右键点击任务栏中的 Docker 图标
2. 选择 "Settings"
3. 点击左侧菜单中的 "Docker Engine"
4. 也可以直接修改配置文件，位置通常在：`C:\Users\<用户名>\.docker\daemon.json`
5. 将以下配置复制到配置编辑器中（注意保留其他已有配置）：

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
```

6. 点击 "Apply & Restart" 按钮应用更改并重启 Docker

## 验证配置

配置完成后，可以通过以下命令验证镜像源是否生效：

```bash
docker info
```

在输出信息中应该能看到 "Registry Mirrors" 部分列出了你配置的镜像源。

## 常见问题

1. 如果遇到权限问题，请确保你有足够的权限修改 Docker 配置文件。

2. 配置完成后如果 Docker 无法启动，请检查配置文件格式是否正确，确保 JSON 格式有效。