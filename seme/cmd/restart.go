package cmd

import (
	"github.com/spf13/cobra"
)

var (
	force              bool
	backendOnlyRestart bool
)

// restartCmd represents the restart command
var restartCmd = &cobra.Command{
	Use:   "restart",
	Short: "Restart Second-Me services",
	Long:  `Stop and restart Second-Me services, including backend and frontend.`,
	Run:   restartServices,
}

func init() {
	RootCmd.AddCommand(restartCmd)
	restartCmd.Flags().BoolVar(&force, "force", false, "Force restart, killing all related processes")
	restartCmd.Flags().BoolVar(&backendOnlyRestart, "backend-only", false, "Restart backend service only")
}

// restartServices is the handler for the restart command
func restartServices(cmd *cobra.Command, args []string) {
	logSection(restartEmoji + " RESTARTING SERVICES")

	// Execute stop command
	logInfo(stopEmoji + " Stopping services...")
	stopCmd.Run(cmd, args)

	// Execute start command, with backend-only flag if specified
	logInfo(startEmoji + " Starting services...")
	if backendOnlyRestart {
		backendOnly = true
	}
	startCmd.Run(cmd, args)

	logSuccess("Services restarted successfully " + successEmoji)
}
