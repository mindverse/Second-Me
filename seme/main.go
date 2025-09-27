package main

import (
	"fmt"
	"os"

	"github.com/bitliu/Second-Me/seme/cmd"
)

// main is the entry point for the application
func main() {
	if err := cmd.RootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
