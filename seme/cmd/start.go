package cmd

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var (
	backendOnly bool
)

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start Second-Me services",
	Long:  `Start Second-Me backend and frontend services, with the option to start backend only.`,
	Run:   startServices,
}

func init() {
	RootCmd.AddCommand(startCmd)
	startCmd.Flags().BoolVar(&backendOnly, "backend-only", false, "Start backend service only")
}

// Check basic configuration and environment
func checkSetupComplete() bool {
	logInfo("Checking if setup is complete...")

	// Check conda environment, simplified to check if environment directory exists
	condaEnv := getEnvOrDefault("CONDA_DEFAULT_ENV", "second-me")
	logInfo(fmt.Sprintf("Using conda environment: %s", bold(condaEnv)))

	// In a real project, should check if conda environment exists
	// Here, simplified to check frontend dependencies directory
	frontendNodeModules := filepath.Join(basePath, "lpm_frontend", "node_modules")
	if !backendOnly && !fileExists(frontendNodeModules) {
		logError("Frontend dependencies not installed. Please run 'make setup' first.")
		return false
	}

	logSuccess("Setup check passed " + checkEmoji)
	return true
}

// Check if file exists
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

// Check if ports are available
func checkPorts() bool {
	logInfo("Checking port availability...")

	// Get port configuration
	backendPort := getEnvOrDefault("LOCAL_APP_PORT", "8002")
	frontendPort := getEnvOrDefault("LOCAL_FRONTEND_PORT", "3000")

	// Check backend port
	if isPortInUse(backendPort) {
		logError(fmt.Sprintf("Backend port %s is already in use!", bold(backendPort)))
		return false
	}

	// If not backend-only mode, check frontend port
	if !backendOnly && isPortInUse(frontendPort) {
		logError(fmt.Sprintf("Frontend port %s is already in use!", bold(frontendPort)))
		return false
	}

	logSuccess("All ports are available " + checkEmoji)
	return true
}

// Check backend health
func checkBackendHealth() bool {
	port := getEnvOrDefault("LOCAL_APP_PORT", "8002")
	url := fmt.Sprintf("http://127.0.0.1:%s/health", port)

	client := http.Client{
		Timeout: 3 * time.Second,
	}

	if verbose {
		logInfo(fmt.Sprintf("Checking backend health at %s", url))
	}

	resp, err := client.Get(url)
	if err != nil {
		if verbose {
			logInfo(fmt.Sprintf("Health check failed: %v", err))
		}
		return false
	}
	defer resp.Body.Close()

	if verbose {
		logInfo(fmt.Sprintf("Health check response: %d", resp.StatusCode))
	}

	return resp.StatusCode == http.StatusOK
}

// Check if frontend is ready
func checkFrontendReady() bool {
	// Try to determine from frontend log if ready
	// In a real project, could verify by checking port or HTTP request
	frontendLogFile := filepath.Join(basePath, LogsDir, "frontend.log")

	// Check if log file exists
	if !fileExists(frontendLogFile) {
		return false
	}

	// Look for "Local:" marker, indicating frontend service is started
	output, err := runCommand(fmt.Sprintf("grep -q 'Local:' %s", frontendLogFile))
	if err != nil {
		return false
	}

	return output != ""
}

// Start backend service
func startBackendService() bool {
	logInfo(startEmoji + " Starting backend service...")

	// Ensure directories exist
	ensureDir(filepath.Join(basePath, LogsDir))
	ensureDir(filepath.Join(basePath, RunDir))

	// Build start command
	startLocalScript := filepath.Join(basePath, ScriptsDir, "start_local.sh")
	logFile := filepath.Join(basePath, LogsDir, "start.log")
	pidFile := getPidFilePath("backend")

	// Check if backend service is already running
	if fileExists(pidFile) {
		pid, err := readPidFile(pidFile)
		if err == nil && isProcessRunning(pid) {
			logWarning(fmt.Sprintf("Backend service is already running with PID: %s", bold(fmt.Sprintf("%d", pid))))

			// Check if it's responding to health checks
			if checkBackendHealth() {
				logSuccess("Backend service is already running and responding " + checkEmoji)
				return true
			} else {
				logWarning("Backend service is running but not responding to health checks, will restart it")
				killProcess(pid, true)
				os.Remove(pidFile)
			}
		} else {
			// PID file exists but process is not running, remove stale PID file
			os.Remove(pidFile)
		}
	}

	// Get conda environment
	condaEnv := getEnvOrDefault("CONDA_DEFAULT_ENV", "second-me")

	// Start backend service with conda environment activated
	// First try to find conda.sh for proper activation
	condaShPath := "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
	if !fileExists(condaShPath) {
		// Try alternative paths
		possiblePaths := []string{
			"/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh",
			"/opt/miniconda3/etc/profile.d/conda.sh",
			"/opt/anaconda3/etc/profile.d/conda.sh",
			"/Users/$(whoami)/miniconda3/etc/profile.d/conda.sh",
			"/Users/$(whoami)/anaconda3/etc/profile.d/conda.sh",
		}

		for _, path := range possiblePaths {
			expandedPath, err := runCommand(fmt.Sprintf("echo %s", path))
			if err == nil && fileExists(strings.TrimSpace(expandedPath)) {
				condaShPath = strings.TrimSpace(expandedPath)
				break
			}
		}
	}

	// Build command with conda activation
	cmdString := ""
	if fileExists(condaShPath) {
		cmdString = fmt.Sprintf("source %s && conda activate %s && %s", condaShPath, condaEnv, startLocalScript)
	} else {
		// Fallback without conda.sh
		cmdString = fmt.Sprintf("conda activate %s && %s", condaEnv, startLocalScript)
	}

	// Start the process
	err := startBackgroundProcess(cmdString, logFile, pidFile)
	if err != nil {
		logError(fmt.Sprintf("Failed to start backend service: %v", err))
		return false
	}

	pid, _ := readPidFile(pidFile)
	logInfo(fmt.Sprintf("Backend service started in background with PID: %s", bold(fmt.Sprintf("%d", pid))))

	// Wait for backend service to be ready
	verbose = true
	if !waitForService(checkBackendHealth, 60, "backend service") {
		// Check backend log for errors when health check fails
		backendLog := filepath.Join(basePath, LogsDir, "backend.log")
		if fileExists(backendLog) {
			// Get last 10 lines of log
			logOutput, err := runCommand(fmt.Sprintf("tail -n 20 %s", backendLog))
			if err == nil && logOutput != "" {
				logWarning("Last 20 lines of backend log:")
				fmt.Println(yellow(logOutput))
			}
		}

		logError("Backend service failed to become ready within 60 seconds")
		logInfo("You can check the backend logs for more information:")
		fmt.Printf("     %s %s\n", magenta("▶"), cyan(fmt.Sprintf("cat %s/logs/backend.log", basePath)))
		fmt.Printf("     %s %s\n", magenta("▶"), cyan(fmt.Sprintf("cat %s/logs/start.log", basePath)))
		return false
	}
	verbose = false

	logSuccess("Backend service is ready " + checkEmoji)
	return true
}

// Start frontend service
func startFrontendService() bool {
	frontendDir := filepath.Join(basePath, "lpm_frontend")

	// Check if frontend directory exists
	if !fileExists(frontendDir) {
		logError("Frontend directory 'lpm_frontend' not found!")
		return false
	}

	logInfo(frontendEmoji + " Starting frontend service...")

	// Check and install dependencies
	nodeModulesDir := filepath.Join(frontendDir, "node_modules")
	if !fileExists(nodeModulesDir) {
		logInfo("Installing frontend dependencies...")
		cmd := fmt.Sprintf("cd %s && npm install", frontendDir)
		if err := runCommandWithOutput(cmd); err != nil {
			logError(fmt.Sprintf("Failed to install frontend dependencies: %v", err))
			return false
		}
		logSuccess("Frontend dependencies installed " + checkEmoji)
	}

	// Start frontend service
	logInfo("Starting frontend dev server...")
	command := fmt.Sprintf("cd %s && npm run dev", frontendDir)
	logFile := filepath.Join(basePath, LogsDir, "frontend.log")
	pidFile := getPidFilePath("frontend")

	if err := startBackgroundProcess(command, logFile, pidFile); err != nil {
		logError(fmt.Sprintf("Failed to start frontend service: %v", err))
		return false
	}

	pid, _ := readPidFile(pidFile)
	logInfo(fmt.Sprintf("Frontend service started in background with PID: %s", bold(fmt.Sprintf("%d", pid))))

	// Wait for frontend service to be ready
	if !waitForService(checkFrontendReady, 120, "frontend service") {
		logError("Frontend service failed to become ready within 120 seconds")
		return false
	}

	logSuccess("Frontend service is ready " + checkEmoji)

	// Display frontend access URL
	port := getEnvOrDefault("LOCAL_FRONTEND_PORT", "3000")
	fmt.Printf("\n  %s  %s\n\n", frontendEmoji, cyan(underline(fmt.Sprintf("Frontend service can be accessed at http://localhost:%s", port))))

	return true
}

// startServices is the handler for the start command
func startServices(cmd *cobra.Command, args []string) {
	logSection(startEmoji + " STARTING SERVICES")

	// Check if setup is complete
	if !checkSetupComplete() {
		os.Exit(1)
	}

	// Check if ports are available
	if !checkPorts() {
		os.Exit(1)
	}

	// Start backend service
	if !startBackendService() {
		os.Exit(1)
	}

	// If not backend-only mode, start frontend service
	if !backendOnly {
		if !startFrontendService() {
			os.Exit(1)
		}
	}

	logSuccess("All services started successfully " + successEmoji)
}
