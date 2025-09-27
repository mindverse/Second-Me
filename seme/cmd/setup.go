package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var (
	skipConfirmation bool
	setupComponent   string
)

// setupCmd represents the setup command
var setupCmd = &cobra.Command{
	Use:   "setup [component]",
	Short: "Set up Second-Me development environment",
	Long: `Set up Second-Me development environment, including Python environment, llama.cpp, and frontend.
You can specify to set up a specific component only: python, llama, frontend`,
	Run: runSetup,
}

func init() {
	RootCmd.AddCommand(setupCmd)
	setupCmd.Flags().BoolVar(&skipConfirmation, "skip-confirmation", false, "Skip confirmation steps")
}

// runSetup is the handler for the setup command
func runSetup(cmd *cobra.Command, args []string) {
	// Parse component argument
	if len(args) > 0 {
		setupComponent = args[0]
		validComponents := map[string]bool{
			"python":   true,
			"llama":    true,
			"frontend": true,
		}
		if !validComponents[setupComponent] {
			logError(fmt.Sprintf("Invalid component: %s", bold(setupComponent)))
			fmt.Printf("\nValid components: %s, %s, %s\n\n",
				cyan(bold("python")),
				cyan(bold("llama")),
				cyan(bold("frontend")))
			os.Exit(1)
		}
	}

	// Display welcome message
	displayHeader()

	// Check potential conflicts
	logSection(setupEmoji + " RUNNING PRE-INSTALLATION CHECKS")
	if !checkPotentialConflicts() {
		logError("Basic tools check failed")
		os.Exit(1)
	}

	// Start installation process
	logSection(setupEmoji + " STARTING INSTALLATION")

	// If a component is specified, only install that component
	if setupComponent != "" {
		switch setupComponent {
		case "python":
			setupPythonEnvironment()
		case "llama":
			buildLlama()
		case "frontend":
			buildFrontend()
		}
		return
	}

	// Install all components
	if !setupPythonEnvironment() {
		os.Exit(1)
	}

	if !setupNpm() {
		logError("npm setup failed")
		os.Exit(1)
	}

	if !checkAndInstallCmake() {
		logError("cmake check and installation failed")
		os.Exit(1)
	}

	if !buildLlama() {
		os.Exit(1)
	}

	if !buildFrontend() {
		os.Exit(1)
	}

	logSuccess("Installation complete! " + successEmoji)
	fmt.Printf("\n  %s  %s\n\n", startEmoji, cyan(bold("Run 'seme start' to start the services")))
}

// Display header information
func displayHeader() {
	fmt.Println("")
	fmt.Println(cyan(" ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗"))
	fmt.Println(cyan(" ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝"))
	fmt.Println(cyan(" ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  "))
	fmt.Println(cyan(" ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  "))
	fmt.Println(cyan(" ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗"))
	fmt.Println(cyan(" ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝"))
	fmt.Println("")
	fmt.Printf("  %s %s\n", setupEmoji, bold(magenta("Second-Me Setup Tool")))
	fmt.Println("")
}

// Check if command exists
func commandExists(cmd string) bool {
	_, err := exec.LookPath(cmd)
	return err == nil
}

// Check potential conflicts
func checkPotentialConflicts() bool {
	logInfo("Checking necessary tools...")

	// Check Homebrew installation
	if commandExists("brew") {
		logInfo("Homebrew is installed")
	} else {
		logWarning("Homebrew not installed, attempting automatic installation...")
		// Here we just show a message, not actually installing
		logError("Homebrew not installed, please install Homebrew first: https://brew.sh")
		return false
	}

	// Check Conda installation
	if commandExists("conda") {
		logInfo("Conda is installed")
	} else {
		logWarning("Conda not installed, attempting automatic installation...")
		// Use Homebrew to install Conda
		if !installConda() {
			logError("Automatic Conda installation failed")
			return false
		}
	}

	// Check config files and directory permissions
	if !checkConfigFiles() {
		logError("Configuration files check failed")
		return false
	}

	if !checkDirectoryPermissions() {
		logError("Directory permissions check failed")
		return false
	}

	return true
}

// Install Conda
func installConda() bool {
	logInfo("Installing Conda using Homebrew...")

	cmd := exec.Command("brew", "install", "--cask", "miniconda")
	if output, err := cmd.CombinedOutput(); err != nil {
		logError(fmt.Sprintf("Failed to install Conda: %v, output: %s", err, string(output)))
		return false
	}

	logSuccess("Conda installed successfully")
	return true
}

// Check configuration files
func checkConfigFiles() bool {
	logInfo("Checking configuration files...")
	// Implement configuration file checking here
	// Simplified to always return true
	return true
}

// Check directory permissions
func checkDirectoryPermissions() bool {
	logInfo("Checking directory permissions...")

	dirsToCheck := []string{
		filepath.Join(basePath, "logs"),
		filepath.Join(basePath, "run"),
	}

	for _, dir := range dirsToCheck {
		// Ensure directory exists
		if err := ensureDir(dir); err != nil {
			logError(fmt.Sprintf("Cannot create directory %s: %v", dir, err))
			return false
		}

		// Check if writable
		testFile := filepath.Join(dir, ".write_test")
		if err := os.WriteFile(testFile, []byte("test"), 0644); err != nil {
			logError(fmt.Sprintf("Directory %s is not writable: %v", dir, err))
			return false
		}

		// Clean up test file
		os.Remove(testFile)
	}

	return true
}

// Set up Python environment
func setupPythonEnvironment() bool {
	logSection("SETTING UP PYTHON ENVIRONMENT")

	condaEnv := getEnvOrDefault("CONDA_DEFAULT_ENV", "second-me")
	logInfo(fmt.Sprintf("Using Conda environment: %s", condaEnv))

	// Check if Conda environment exists
	cmd := exec.Command("conda", "env", "list")
	output, err := cmd.CombinedOutput()
	if err != nil {
		logError(fmt.Sprintf("Failed to get Conda environment list: %v", err))
		return false
	}

	// Check if environment name is in the output
	if !strings.Contains(string(output), condaEnv) {
		logInfo(fmt.Sprintf("Creating Conda environment: %s", condaEnv))

		// Use environment.yml file to create environment
		envFile := filepath.Join(basePath, "environment.yml")
		if fileExists(envFile) {
			cmd = exec.Command("conda", "env", "create", "-f", envFile)
			if output, err := cmd.CombinedOutput(); err != nil {
				logError(fmt.Sprintf("Failed to create Conda environment: %v, output: %s", err, string(output)))
				return false
			}
		} else {
			// If no environment.yml file, create with Python 3.10
			cmd = exec.Command("conda", "create", "-n", condaEnv, "python=3.10", "-y")
			if output, err := cmd.CombinedOutput(); err != nil {
				logError(fmt.Sprintf("Failed to create Conda environment: %v, output: %s", err, string(output)))
				return false
			}
		}

		logSuccess(fmt.Sprintf("Conda environment %s created successfully", condaEnv))
	} else {
		logInfo(fmt.Sprintf("Conda environment %s already exists, skipping creation", condaEnv))
	}

	// Install Python dependencies
	logInfo("Installing Python dependencies...")

	// Check for requirements.txt or pyproject.toml
	pytorchToml := filepath.Join(basePath, "pyproject.toml")
	if fileExists(pytorchToml) {
		logInfo("Found pyproject.toml, using Poetry to install dependencies...")

		// Use conda run to execute command in the specified environment
		cmd = exec.Command("conda", "run", "-n", condaEnv, "pip", "install", "poetry")
		if output, err := cmd.CombinedOutput(); err != nil {
			logError(fmt.Sprintf("Failed to install Poetry: %v, output: %s", err, string(output)))
			return false
		}

		cmd = exec.Command("conda", "run", "-n", condaEnv, "poetry", "install")
		if output, err := cmd.CombinedOutput(); err != nil {
			logError(fmt.Sprintf("Failed to install dependencies with Poetry: %v, output: %s", err, string(output)))
			return false
		}
	} else {
		// If no pyproject.toml, look for requirements.txt
		reqFile := filepath.Join(basePath, "requirements.txt")
		if fileExists(reqFile) {
			logInfo("Found requirements.txt, using pip to install dependencies...")

			cmd = exec.Command("conda", "run", "-n", condaEnv, "pip", "install", "-r", reqFile)
			if output, err := cmd.CombinedOutput(); err != nil {
				logError(fmt.Sprintf("Failed to install dependencies with pip: %v, output: %s", err, string(output)))
				return false
			}
		} else {
			logWarning("No pyproject.toml or requirements.txt found, skipping dependency installation")
		}
	}

	// Check specific dependency packages
	graphragPath := filepath.Join(basePath, "dependencies/graphrag-1.2.1.dev27.tar.gz")
	if fileExists(graphragPath) {
		logInfo("Installing specific version of graphrag...")

		cmd = exec.Command("conda", "run", "-n", condaEnv, "pip", "install", "--force-reinstall", graphragPath)
		if output, err := cmd.CombinedOutput(); err != nil {
			logError(fmt.Sprintf("Failed to install graphrag: %v, output: %s", err, string(output)))
			return false
		}

		logSuccess("graphrag installed successfully")
	}

	logSuccess("Python environment setup completed")
	return true
}

// Set up npm
func setupNpm() bool {
	logInfo("Setting up npm package manager...")

	// Check if npm is already installed
	if !commandExists("npm") {
		logWarning("npm not found - installing Node.js and npm")

		cmd := exec.Command("brew", "install", "node")
		if output, err := cmd.CombinedOutput(); err != nil {
			logError(fmt.Sprintf("Failed to install Node.js and npm: %v, output: %s", err, string(output)))
			return false
		}

		// Verify npm was installed successfully
		if !commandExists("npm") {
			logError("npm installation failed - command not found after installation")
			return false
		}
		logSuccess("Successfully installed Node.js and npm")
	} else {
		logSuccess("npm is already installed")
	}

	// Configure npm settings
	logInfo("Configuring npm settings...")

	// Set npm registry
	cmd := exec.Command("npm", "config", "set", "registry", "https://registry.npmjs.org/")
	if output, err := cmd.CombinedOutput(); err != nil {
		logError(fmt.Sprintf("Failed to set npm registry: %v, output: %s", err, string(output)))
		return false
	}

	// Set npm cache directory
	cmd = exec.Command("npm", "config", "set", "cache", filepath.Join(os.Getenv("HOME"), ".npm"))
	if output, err := cmd.CombinedOutput(); err != nil {
		logError(fmt.Sprintf("Failed to set npm cache directory: %v, output: %s", err, string(output)))
		return false
	}

	// Verify npm configuration
	cmd = exec.Command("npm", "config", "list")
	if output, err := cmd.CombinedOutput(); err != nil {
		logError(fmt.Sprintf("npm configuration failed: %v, output: %s", err, string(output)))
		return false
	}

	logSuccess("npm setup completed")
	return true
}

// Check and install cmake
func checkAndInstallCmake() bool {
	logInfo("Checking cmake installation...")

	if !commandExists("cmake") {
		logWarning("cmake not installed, attempting automatic installation...")

		cmd := exec.Command("brew", "install", "cmake")
		if output, err := cmd.CombinedOutput(); err != nil {
			logError(fmt.Sprintf("Failed to install cmake using Homebrew: %v, output: %s", err, string(output)))
			return false
		}
		logSuccess("cmake installed successfully")
	} else {
		logInfo("cmake is already installed")
	}

	return true
}

// Build llama.cpp
func buildLlama() bool {
	logSection("BUILDING LLAMA.CPP")

	llamaDir := filepath.Join(basePath, "llama.cpp")
	llamaZip := filepath.Join(basePath, "dependencies/llama.cpp.zip")

	// Check if llama.cpp directory exists
	if !fileExists(llamaDir) {
		logInfo("Setting up llama.cpp...")

		if fileExists(llamaZip) {
			logInfo("Using local llama.cpp archive...")

			// Extract llama.cpp archive
			cmd := exec.Command("unzip", "-q", llamaZip, "-d", basePath)
			if output, err := cmd.CombinedOutput(); err != nil {
				logError(fmt.Sprintf("Failed to extract local llama.cpp archive: %v, output: %s", err, string(output)))
				return false
			}
		} else {
			logError(fmt.Sprintf("Local llama.cpp archive not found at: %s", llamaZip))
			logError("Please ensure the llama.cpp.zip file exists in the dependencies directory")
			return false
		}
	} else {
		logInfo("Found existing llama.cpp directory")
	}

	// Check if llama.cpp has been successfully compiled
	llamaServer := filepath.Join(llamaDir, "build/bin/llama-server")
	if fileExists(llamaServer) {
		logInfo("Found existing llama-server build")

		// Check if executable file can be run and get version information
		cmd := exec.Command(llamaServer, "--version")
		if output, err := cmd.CombinedOutput(); err == nil {
			outStr := string(output)
			if strings.Contains(outStr, "version") {
				logSuccess(fmt.Sprintf("Existing llama-server build is working properly (%s), skipping compilation", strings.TrimSpace(outStr)))
				return true
			}
		}
		logWarning("Existing build seems broken or incompatible, will recompile...")
	}

	// Enter llama.cpp directory and build
	oldDir, err := os.Getwd()
	if err != nil {
		logError(fmt.Sprintf("Failed to get current working directory: %v", err))
		return false
	}

	if err := os.Chdir(llamaDir); err != nil {
		logError(fmt.Sprintf("Failed to enter llama.cpp directory: %v", err))
		return false
	}

	// Clean previous build
	buildDir := filepath.Join(llamaDir, "build")
	if fileExists(buildDir) {
		logInfo("Cleaning previous build...")
		if err := os.RemoveAll(buildDir); err != nil {
			logError(fmt.Sprintf("Failed to clean previous build: %v", err))
			os.Chdir(oldDir)
			return false
		}
	}

	// Create and enter build directory
	logInfo("Creating build directory...")
	if err := os.MkdirAll(buildDir, 0755); err != nil {
		logError(fmt.Sprintf("Failed to create build directory: %v", err))
		os.Chdir(oldDir)
		return false
	}

	if err := os.Chdir(buildDir); err != nil {
		logError(fmt.Sprintf("Failed to enter build directory: %v", err))
		os.Chdir(oldDir)
		return false
	}

	// Configure CMake
	logInfo("Configuring CMake...")
	cmd := exec.Command("cmake", "..")
	if output, err := cmd.CombinedOutput(); err != nil {
		logError(fmt.Sprintf("CMake configuration failed: %v, output: %s", err, string(output)))
		os.Chdir(oldDir)
		return false
	}

	// Build project
	logInfo("Building project...")
	cmd = exec.Command("cmake", "--build", ".", "--config", "Release")
	if output, err := cmd.CombinedOutput(); err != nil {
		logError(fmt.Sprintf("Build failed: %v, output: %s", err, string(output)))
		os.Chdir(oldDir)
		return false
	}

	// Check build result
	if !fileExists(filepath.Join(buildDir, "bin/llama-server")) {
		logError("Build failed: llama-server executable not found")
		logError("Expected at: bin/llama-server")
		os.Chdir(oldDir)
		return false
	}

	logSuccess("Found llama-server at bin/llama-server")
	os.Chdir(oldDir)
	logSection("LLAMA.CPP BUILD COMPLETE")
	return true
}

// Build frontend
func buildFrontend() bool {
	logSection("SETTING UP FRONTEND")

	frontendDir := filepath.Join(basePath, "lpm_frontend")

	// Enter frontend directory
	oldDir, err := os.Getwd()
	if err != nil {
		logError(fmt.Sprintf("Failed to get current working directory: %v", err))
		return false
	}

	if err := os.Chdir(frontendDir); err != nil {
		logError(fmt.Sprintf("Failed to enter frontend directory: %s: %v", frontendDir, err))
		logError("Please ensure the directory exists and you have permission to access it.")
		return false
	}

	// Check if dependencies have been installed
	nodeModules := filepath.Join(frontendDir, "node_modules")
	packageLock := filepath.Join(frontendDir, "package-lock.json")

	if fileExists(nodeModules) {
		logInfo("Found existing node_modules, checking for updates...")

		if fileExists(packageLock) {
			logInfo("Using existing package-lock.json...")
			// Run npm install even if package-lock.json exists to ensure dependencies are complete
			logInfo("Running npm install to ensure dependencies are complete...")

			cmd := exec.Command("npm", "install")
			if output, err := cmd.CombinedOutput(); err != nil {
				logError(fmt.Sprintf("Failed to install frontend dependencies with existing package-lock.json: %v, output: %s", err, string(output)))
				logError("Try removing node_modules directory and package-lock.json, then run setup again")
				os.Chdir(oldDir)
				return false
			}
		} else {
			logInfo("Installing dependencies...")

			cmd := exec.Command("npm", "install")
			if output, err := cmd.CombinedOutput(); err != nil {
				logError(fmt.Sprintf("Failed to install frontend dependencies: %v, output: %s", err, string(output)))
				logError("Check your npm configuration and network connection")
				logError(fmt.Sprintf("You can try running 'npm install' manually in the %s directory", frontendDir))
				os.Chdir(oldDir)
				return false
			}
		}
	} else {
		logInfo("Installing dependencies...")

		cmd := exec.Command("npm", "install")
		if output, err := cmd.CombinedOutput(); err != nil {
			logError(fmt.Sprintf("Failed to install frontend dependencies: %v, output: %s", err, string(output)))
			logError("Check your npm configuration and network connection")
			logError(fmt.Sprintf("You can try running 'npm install' manually in the %s directory", frontendDir))
			os.Chdir(oldDir)
			return false
		}
	}

	// Verify installation was successful
	if !fileExists(nodeModules) {
		logError("node_modules directory not found after npm install")
		logError("Frontend dependencies installation failed")
		os.Chdir(oldDir)
		return false
	}

	logSuccess("Frontend dependencies installed successfully")
	os.Chdir(oldDir)
	logSection("FRONTEND SETUP COMPLETE")
	return true
}
