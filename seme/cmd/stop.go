package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// stopCmd represents the stop command
var stopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop Second-Me services",
	Long:  `Stop all running Second-Me services, including backend, frontend, and related processes.`,
	Run:   stopServices,
}

func init() {
	RootCmd.AddCommand(stopCmd)
}

// Stop a specific process
func stopProcess(pidFile string, description string) {
	if fileExists(pidFile) {
		pid, err := readPidFile(pidFile)
		if err != nil {
			logWarning(fmt.Sprintf("Failed to read PID file: %v", err))
			return
		}

		if isProcessRunning(pid) {
			logInfo(fmt.Sprintf("Stopping %s process (PID: %s)...", bold(description), bold(fmt.Sprintf("%d", pid))))
			if err := killProcess(pid, false); err != nil {
				// 检查错误信息，判断进程是否已经结束
				if strings.Contains(err.Error(), "process already finished") {
					logInfo(fmt.Sprintf("Process %d was already finished", pid))
				} else {
					logWarning(fmt.Sprintf("Process still running, trying force termination..."))
					if err := killProcess(pid, true); err != nil {
						if strings.Contains(err.Error(), "process already finished") {
							logInfo(fmt.Sprintf("Process %d was already finished", pid))
						} else {
							logError(fmt.Sprintf("Cannot terminate process: %v", err))
						}
					}
				}
			}
			logSuccess(fmt.Sprintf("%s process stopped (PID: %s) %s", bold(description), bold(fmt.Sprintf("%d", pid)), stopEmoji))
		} else {
			logInfo(fmt.Sprintf("%s process not running", description))
		}

		// Remove PID file
		if err := os.Remove(pidFile); err != nil {
			logWarning(fmt.Sprintf("Cannot remove PID file: %v", err))
		}
	}
}

// Check and close process on a specific port
func stopProcessOnPort(port string, description string) {
	logInfo(fmt.Sprintf("Checking process on port %s...", bold(port)))

	// Use lsof to find process using the port
	cmd := fmt.Sprintf("lsof -ti:%s", port)
	output, err := runCommand(cmd)
	if err != nil || output == "" {
		logInfo(fmt.Sprintf("No process running on port %s", bold(port)))
		return
	}

	// 修改: 处理可能有多个PID的情况
	pidStrings := strings.Split(strings.TrimSpace(output), "\n")
	logInfo(fmt.Sprintf("Found %d processes on port %s", len(pidStrings), bold(port)))

	for _, pidStr := range pidStrings {
		// 去除每个PID字符串的空白
		pidStr = strings.TrimSpace(pidStr)
		if pidStr == "" {
			continue
		}

		// 解析PID
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			logWarning(fmt.Sprintf("Cannot parse process ID: %v, value: %s", err, pidStr))
			continue
		}

		logInfo(fmt.Sprintf("Stopping process using port %s (PID: %s)...", bold(port), bold(fmt.Sprintf("%d", pid))))
		if err := killProcess(pid, true); err != nil {
			logError(fmt.Sprintf("Cannot terminate process: %v", err))
		} else {
			logSuccess(fmt.Sprintf("Process on port %s (PID: %s) terminated %s", bold(port), bold(fmt.Sprintf("%d", pid)), stopEmoji))
		}
	}

	// 检查是否所有进程都已终止
	time.Sleep(500 * time.Millisecond) // 给进程一些时间终止

	remainingOutput, _ := runCommand(cmd)
	if remainingOutput != "" {
		pidCount := len(strings.Split(strings.TrimSpace(remainingOutput), "\n"))
		logWarning(fmt.Sprintf("Still %d processes running on port %s. You might need to manually kill them.", pidCount, bold(port)))
	} else {
		logSuccess(fmt.Sprintf("All processes on port %s stopped successfully %s", bold(port), stopEmoji))
	}
}

// Stop processes matching a specific pattern
func stopProcessesByPattern(pattern string, description string) {
	logInfo(fmt.Sprintf("Checking %s processes...", bold(description)))

	pids, err := findProcessByPattern(pattern)
	if err != nil {
		logWarning(fmt.Sprintf("Failed to find processes: %v", err))
		return
	}

	if len(pids) == 0 {
		logInfo(fmt.Sprintf("No %s processes found", description))
		return
	}

	for _, pid := range pids {
		logInfo(fmt.Sprintf("Stopping %s process (PID: %s)...", bold(description), bold(fmt.Sprintf("%d", pid))))
		if err := killProcess(pid, false); err != nil {
			// 检查错误信息，判断进程是否已经结束
			if strings.Contains(err.Error(), "process already finished") {
				logInfo(fmt.Sprintf("Process %d was already finished", pid))
			} else {
				logWarning(fmt.Sprintf("Process still running, trying force termination..."))
				if err := killProcess(pid, true); err != nil {
					if strings.Contains(err.Error(), "process already finished") {
						logInfo(fmt.Sprintf("Process %d was already finished", pid))
					} else {
						logError(fmt.Sprintf("Cannot terminate process: %v", err))
					}
				}
			}
		}
		time.Sleep(500 * time.Millisecond)
	}

	logSuccess(fmt.Sprintf("%s processes stopped %s", bold(description), stopEmoji))
}

// stopServices is the handler for the stop command
func stopServices(cmd *cobra.Command, args []string) {
	logSection(stopEmoji + " STOPPING SERVICES")

	// Get port configuration
	backendPort := getEnvOrDefault("LOCAL_APP_PORT", "8002")
	frontendPort := getEnvOrDefault("LOCAL_FRONTEND_PORT", "3000")

	// Create run directory if it doesn't exist
	ensureDir(filepath.Join(basePath, RunDir))

	// Stop backend service
	backendPidFile := getPidFilePath("backend")
	stopProcess(backendPidFile, "backend")

	// Check backend port
	stopProcessOnPort(backendPort, "backend")

	// Stop llama-server processes
	stopProcessesByPattern("llama-server", "llama-server")

	// Check port 8080 (commonly used for llama-server)
	stopProcessOnPort("8080", "llama-server")

	// Stop frontend service
	frontendPidFile := getPidFilePath("frontend")
	stopProcess(frontendPidFile, "frontend")

	// Stop all Next.js related processes
	stopProcessesByPattern("next dev|next-server", "Next.js")

	// Stop all npm run dev related processes
	stopProcessesByPattern("npm run dev", "npm")

	// Check frontend port
	stopProcessOnPort(frontendPort, "frontend")

	logSuccess("All services stopped successfully " + successEmoji)
}
