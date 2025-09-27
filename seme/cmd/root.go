package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/fatih/color"
	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var (
	// Global flags
	verbose  bool
	envFile  string
	basePath string

	// Color definitions
	red       = color.New(color.FgRed).SprintFunc()
	green     = color.New(color.FgGreen).SprintFunc()
	yellow    = color.New(color.FgYellow).SprintFunc()
	blue      = color.New(color.FgBlue).SprintFunc()
	cyan      = color.New(color.FgCyan).SprintFunc()
	magenta   = color.New(color.FgMagenta).SprintFunc()
	gray      = color.New(color.FgHiBlack).SprintFunc()
	bold      = color.New(color.Bold).SprintFunc()
	underline = color.New(color.Underline).SprintFunc()

	// Emoji set for different log types
	infoEmoji     = "ℹ️ "
	successEmoji  = "✅ "
	warningEmoji  = "⚠️ "
	errorEmoji    = "❌ "
	sectionEmoji  = "🔷 "
	startEmoji    = "🚀 "
	stopEmoji     = "🛑 "
	restartEmoji  = "🔄 "
	statusEmoji   = "📊 "
	setupEmoji    = "⚙️ "
	pythonEmoji   = "🐍 "
	frontendEmoji = "🌐 "
	llamaEmoji    = "🦙 "
	checkEmoji    = "✓ "
)

// RootCmd represents the base command when called without any subcommands
var RootCmd = &cobra.Command{
	Use:   "seme",
	Short: "Second-Me CLI Tool",
	Long: `seme is a CLI tool for managing Second-Me application,
it can start, stop, restart the application and check the status.`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		// Find project root if not explicitly set
		if basePath == "" {
			cwd, err := os.Getwd()
			if err != nil {
				fmt.Println(red("Cannot get current working directory:"), err)
				os.Exit(1)
			}

			// Look for .env file up the directory tree to identify project root
			dir := cwd
			for {
				if _, err := os.Stat(filepath.Join(dir, ".env")); err == nil {
					basePath = dir
					break
				}
				parent := filepath.Dir(dir)
				if parent == dir {
					break
				}
				dir = parent
			}

			// If .env file not found, use current directory
			if basePath == "" {
				basePath = cwd
			}
		}

		// Load environment variables
		if envFile == "" {
			envFile = filepath.Join(basePath, ".env")
		}

		err := godotenv.Load(envFile)
		if err != nil {
			// Just show a warning as some commands may not require environment variables
			fmt.Println(yellow(warningEmoji+"Warning: Cannot load environment variables file:"), envFile)
		}
	},
}

// Initialize command line flags
func init() {
	RootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose output")
	RootCmd.PersistentFlags().StringVar(&envFile, "env-file", "", "Path to environment variables file (default: .env)")
	RootCmd.PersistentFlags().StringVar(&basePath, "base-path", "", "Project root directory path")
}

// Get current timestamp
func getTimestamp() string {
	return time.Now().Format("2006-01-02 15:04:05")
}

// Log functions
func logInfo(message string) {
	fmt.Printf("%s %s %s\n", gray("["+getTimestamp()+"]"), blue(infoEmoji+"[INFO]"), message)
}

func logSuccess(message string) {
	fmt.Printf("%s %s %s\n", gray("["+getTimestamp()+"]"), green(successEmoji+"[SUCCESS]"), message)
}

func logWarning(message string) {
	fmt.Printf("%s %s %s\n", gray("["+getTimestamp()+"]"), yellow(warningEmoji+"[WARNING]"), message)
}

func logError(message string) {
	fmt.Printf("%s %s %s\n", gray("["+getTimestamp()+"]"), red(errorEmoji+"[ERROR]"), message)
}

func logSection(message string) {
	bar := strings.Repeat("━", 80)
	fmt.Printf("\n%s\n  %s %s\n%s\n\n", cyan(bar), sectionEmoji, cyan(bold(message)), cyan(bar))
}

// Get value from environment variables, or use default value if not exists
func getEnvOrDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

// Check if port is in use
func isPortInUse(port string) bool {
	cmd := exec.Command("lsof", "-i", fmt.Sprintf(":%s", port))
	err := cmd.Run()
	// If command executes successfully, the port is in use
	return err == nil
}

// Get process information by PID file
func getProcessInfo(pidFile string) (int, string, string, bool) {
	// In a real project, implement this function to read the PID file and get process information
	// This is just a placeholder
	return 0, "", "", false
}
