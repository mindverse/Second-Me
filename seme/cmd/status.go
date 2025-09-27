package cmd

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

// statusCmd represents the status command
var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Check Second-Me services status",
	Long:  `Check the running status of Second-Me services, including frontend, backend, and related services.`,
	Run:   checkStatus,
}

func init() {
	RootCmd.AddCommand(statusCmd)
}

// Check process status
func checkProcessStatus(pidFile string, name string) bool {
	fmt.Printf("  %s: ", bold(name))

	if !fileExists(pidFile) {
		fmt.Printf("%s (No PID file)\n", red("Not running ❌"))
		return false
	}

	pid, err := readPidFile(pidFile)
	if err != nil {
		fmt.Printf("%s (Invalid PID file: %v)\n", red("Not running ❌"), err)
		return false
	}

	if !isProcessRunning(pid) {
		fmt.Printf("%s (PID file exists but process is dead: %s)\n", red("Not running ❌"), bold(fmt.Sprintf("%d", pid)))
		return false
	}

	// Get process information
	cmd := fmt.Sprintf("ps -p %d -o comm= 2>/dev/null", pid)
	processName, _ := runCommand(cmd)

	cmd = fmt.Sprintf("ps -p %d -o args= 2>/dev/null", pid)
	processCmd, _ := runCommand(cmd)

	fmt.Printf("%s (PID: %s, Process: %s)\n", green("Running ✅"), bold(fmt.Sprintf("%d", pid)), bold(processName))
	fmt.Printf("     %s %s\n", magenta("▶"), processCmd)

	return true
}

// Check port status
func checkPortStatus(port string, description string) bool {
	fmt.Printf("  %s: ", bold(fmt.Sprintf("Port %s", port)))

	// Check if port is in use
	cmd := fmt.Sprintf("lsof -ti:%s 2>/dev/null", port)
	output, err := runCommand(cmd)
	if err != nil || output == "" {
		fmt.Printf("%s\n", red("Not in use ❌"))
		return false
	}

	// Process may have multiple PIDs - split by newline and take the first one
	pidStr := strings.Split(strings.TrimSpace(output), "\n")[0]

	// Parse PID
	pid, err := strconv.Atoi(pidStr)
	if err != nil {
		fmt.Printf("%s (Cannot parse PID: %v)\n", red("Error ⚠️"), err)
		return false
	}

	// Get process information
	cmd = fmt.Sprintf("ps -p %d -o comm= 2>/dev/null", pid)
	processName, _ := runCommand(cmd)

	cmd = fmt.Sprintf("ps -p %d -o args= 2>/dev/null", pid)
	processCmd, _ := runCommand(cmd)

	// If multiple PIDs, show a note
	countPIDs := len(strings.Split(strings.TrimSpace(output), "\n"))
	if countPIDs > 1 {
		fmt.Printf("%s (PID: %s, Process: %s, +%d more)\n",
			green("In use ✅"),
			bold(fmt.Sprintf("%d", pid)),
			bold(processName),
			countPIDs-1)
	} else {
		fmt.Printf("%s (PID: %s, Process: %s)\n",
			green("In use ✅"),
			bold(fmt.Sprintf("%d", pid)),
			bold(processName))
	}

	fmt.Printf("     %s %s\n", magenta("▶"), processCmd)

	return true
}

// Create status summary
func printStatusSummary(backendRunning, frontendRunning, port8080Running bool) {
	fmt.Printf("\n%s\n", bold("Summary:"))

	if backendRunning {
		fmt.Printf("  %s Backend: %s\n", "🖥️ ", green("Running ✅"))
	} else {
		fmt.Printf("  %s Backend: %s\n", "🖥️ ", red("Not running ❌"))
	}

	if frontendRunning {
		fmt.Printf("  %s Frontend: %s\n", "🌐", green("Running ✅"))
	} else {
		fmt.Printf("  %s Frontend: %s\n", "🌐", red("Not running ❌"))
	}

	if port8080Running {
		fmt.Printf("  %s LLM Server: %s\n", "🦙", green("Running ✅"))
	} else {
		fmt.Printf("  %s LLM Server: %s\n", "🦙", red("Not running ❌"))
	}

	// Add a note about how to start services if they're not running
	if !backendRunning || !frontendRunning || !port8080Running {
		fmt.Printf("\n%s To start services, run: %s\n", infoEmoji, cyan("seme start"))
	}
}

// checkStatus is the handler for the status command
func checkStatus(cmd *cobra.Command, args []string) {
	logSection(statusEmoji + " SERVICE STATUS")

	// Get port configuration
	backendPort := getEnvOrDefault("LOCAL_APP_PORT", "8002")
	frontendPort := getEnvOrDefault("LOCAL_FRONTEND_PORT", "3000")

	// Check backend service status
	fmt.Printf("%s\n", bold("Backend Service:"))
	backendPidFile := getPidFilePath("backend")
	backendPidStatus := checkProcessStatus(backendPidFile, "PID File")

	backendPortStatus := checkPortStatus(backendPort, "Backend Port")

	// Check frontend service status
	fmt.Printf("\n%s\n", bold("Frontend Service:"))
	frontendPidFile := getPidFilePath("frontend")
	frontendPidStatus := checkProcessStatus(frontendPidFile, "PID File")

	frontendPortStatus := checkPortStatus(frontendPort, "Frontend Port")

	// Check port 8080 status (commonly used for llama-server)
	fmt.Printf("\n%s\n", bold("LLM Server Status:"))
	port8080Status := checkPortStatus("8080", "LLaMA Port")

	// Print status summary
	backendRunning := backendPidStatus || backendPortStatus
	frontendRunning := frontendPidStatus || frontendPortStatus
	printStatusSummary(backendRunning, frontendRunning, port8080Status)
}
