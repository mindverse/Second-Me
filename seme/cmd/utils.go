package cmd

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Common project paths and files
const (
	RunDir     = "run"
	LogsDir    = "logs"
	ScriptsDir = "scripts"
)

// Get PID file path
func getPidFilePath(name string) string {
	return filepath.Join(basePath, RunDir, fmt.Sprintf(".%s.pid", name))
}

// Read PID file
func readPidFile(pidFile string) (int, error) {
	data, err := os.ReadFile(pidFile)
	if err != nil {
		return 0, err
	}

	pidStr := strings.TrimSpace(string(data))
	pid, err := strconv.Atoi(pidStr)
	if err != nil {
		return 0, fmt.Errorf("PID file contains invalid PID: %s", pidStr)
	}

	return pid, nil
}

// Write PID file
func writePidFile(pidFile string, pid int) error {
	// Ensure directory exists
	dir := filepath.Dir(pidFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	return os.WriteFile(pidFile, []byte(strconv.Itoa(pid)), 0644)
}

// Check if process exists
func isProcessRunning(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}

	// On Unix systems, FindProcess always succeeds, need to send signal 0 to check if process exists
	err = process.Signal(syscall.Signal(0))
	return err == nil
}

// Terminate process
func killProcess(pid int, force bool) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}

	// 在发送终止信号前，先检查进程是否存在
	if !isProcessRunning(pid) {
		return fmt.Errorf("process already finished")
	}

	var sig syscall.Signal
	if force {
		sig = syscall.SIGKILL
	} else {
		sig = syscall.SIGTERM
	}

	return process.Signal(sig)
}

// Start background process
func startBackgroundProcess(command string, logFile string, pidFile string) error {
	// Check if log file can be created
	logDir := filepath.Dir(logFile)
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return fmt.Errorf("cannot create log directory %s: %v", logDir, err)
	}

	// Check if PID directory can be created
	pidDir := filepath.Dir(pidFile)
	if err := os.MkdirAll(pidDir, 0755); err != nil {
		return fmt.Errorf("cannot create PID directory %s: %v", pidDir, err)
	}

	// Open log file
	file, err := os.Create(logFile)
	if err != nil {
		return fmt.Errorf("cannot create log file: %v", err)
	}
	defer file.Close()

	// Prepare command - use zsh instead of sh for better conda compatibility
	cmd := exec.Command("/bin/zsh", "-c", command)
	cmd.Stdout = file
	cmd.Stderr = file
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true, // Put the process in a new process group
	}

	// Start process
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cannot start process: %v", err)
	}

	// Write PID file
	if err := writePidFile(pidFile, cmd.Process.Pid); err != nil {
		// If cannot write PID, terminate process
		cmd.Process.Kill()
		return fmt.Errorf("cannot write PID file: %v", err)
	}

	return nil
}

// Run command and return output
func runCommand(command string) (string, error) {
	cmd := exec.Command("/bin/zsh", "-c", command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("command execution failed: %v, output: %s", err, string(output))
	}
	return string(output), nil
}

// Run command in foreground, streaming output to console
func runCommandWithOutput(command string) error {
	cmd := exec.Command("/bin/zsh", "-c", command)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("cannot get command stdout: %v", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("cannot get command stderr: %v", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cannot start command: %v", err)
	}

	// Handle stdout
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			fmt.Println(scanner.Text())
		}
	}()

	// Handle stderr
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			fmt.Fprintln(os.Stderr, scanner.Text())
		}
	}()

	return cmd.Wait()
}

// Find process IDs matching pattern
func findProcessByPattern(pattern string) ([]int, error) {
	cmd := exec.Command("/bin/zsh", "-c", fmt.Sprintf("pgrep -f '%s'", pattern))
	output, err := cmd.Output()
	if err != nil {
		// Check if non-zero exit code is due to no matching processes
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			return []int{}, nil
		}
		return nil, err
	}

	var pids []int
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		if line == "" {
			continue
		}
		pid, err := strconv.Atoi(line)
		if err != nil {
			return nil, fmt.Errorf("cannot parse PID: %v", err)
		}
		pids = append(pids, pid)
	}

	return pids, nil
}

// Check if directory exists, create if not
func ensureDir(dir string) error {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return os.MkdirAll(dir, 0755)
	}
	return nil
}

// Wait for service health check
func waitForService(checkFunc func() bool, maxAttempts int, description string) bool {
	logInfo(fmt.Sprintf("Waiting for %s to be ready...", description))

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if checkFunc() {
			return true
		}

		// Always log every 5 attempts regardless of verbose setting
		if verbose || attempt%5 == 0 {
			logInfo(fmt.Sprintf("Attempt %d/%d: %s not ready, waiting...", attempt, maxAttempts, description))
		}

		// Gradually increase wait time between attempts (up to 3 seconds)
		waitTime := time.Duration(min(3, 1+(attempt/10))) * time.Second
		time.Sleep(waitTime)
	}

	return false
}

// Helper function for min since Go < 1.21 doesn't have math.Min for integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
