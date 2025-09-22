# Project Configuration Guide for Environments in mainland China

This document provides methods to configure the project using domestic network resources in mainland China, applicable for both Docker and non-Docker users.

## Non-Docker User Configuration

Non-Docker users can directly use the following command for configuration:

```bash
make setup-cn
```

## Docker User Configuration

Docker users need to modify the Docker configuration file to use mirrors in China. Below are configuration methods for different operating systems:

### Mac System

#### Method 1: Configure through Docker Desktop UI

1. Click on the Docker icon in the top menu bar
2. Select "Settings" or "Preferences"
3. Click on the "Docker Engine" tab
4. Copy the following configuration to the configuration editor (note to preserve any existing configurations):

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

5. Click the "Apply & Restart" button to apply changes and restart Docker

#### Method 2: Directly edit the configuration file

1. Open Terminal
2. Open the Docker configuration file with a text editor:

```bash
mkdir -p ~/.docker
nano ~/.docker/daemon.json
```

3. Copy the following content to the file:

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
```

4. Save the file (in nano, press Ctrl+O, then Enter, then Ctrl+X to exit)
5. Restart Docker Desktop

### Linux System

1. Create or edit the Docker configuration file:

```bash
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
```

2. Copy the following content to the file:

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

3. Save the file and restart the Docker service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Windows System

1. Right-click on the Docker icon in the taskbar
2. Select "Settings"
3. Click on "Docker Engine" in the left menu
4. You can also directly modify the configuration file, typically located at: `C:\Users\<username>\.docker\daemon.json`
5. Copy the following configuration to the configuration editor (note to preserve any existing configurations):

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

6. Click the "Apply & Restart" button to apply changes and restart Docker

## Verify Configuration

After configuration, you can verify if the mirror sources are effective with the following command:

```bash
docker info
```

In the output information, you should be able to see the "Registry Mirrors" section listing the mirror sources you configured.

## Common Issues

1. If you encounter permission issues, please ensure you have sufficient permissions to modify the Docker configuration file.

2. If Docker fails to start after configuration, check if the configuration file format is correct and ensure the JSON format is valid.
