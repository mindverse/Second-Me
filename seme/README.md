# seme - Second-Me CLI Tool

`seme` is a command-line tool written in Golang to manage the lifecycle of Second-Me application services. It replaces the original set of shell scripts with a more consistent interface and better error handling.

## Installation

### Building from Source

```bash
git clone https://github.com/bitliu/Second-Me.git
cd Second-Me/seme
go build -o seme
```

You can then move the compiled binary to a directory in your system path, for example:

```bash
sudo mv seme /usr/local/bin/
```

## Usage

`seme` provides the following commands:

### Setting Up the Environment

```bash
# Set up the complete environment (Python, llama.cpp, and frontend)
seme setup

# Set up specific components only
seme setup python
seme setup llama
seme setup frontend

# Skip confirmation steps
seme setup --skip-confirmation
```

### Starting Services

```bash
# Start all services (frontend and backend)
seme start

# Start backend service only
seme start --backend-only
```

### Stopping Services

```bash
# Stop all services
seme stop
```

### Restarting Services

```bash
# Restart all services
seme restart

# Restart backend service only
seme restart --backend-only

# Force restart (terminate all related processes)
seme restart --force
```

### Checking Service Status

```bash
# Check the status of all services
seme status
```

## Configuration

`seme` will search for a `.env` file in the current directory and its parent directories to load environment variables. You can also specify the location of the environment variable file using command-line arguments:

```bash
seme --env-file /path/to/.env start
```

You can also specify the project root directory:

```bash
seme --base-path /path/to/project start
```

## Global Options

The following options are available for all commands:

- `--base-path string`: Specify the project root directory
- `--env-file string`: Specify the path to the environment variable file
- `-v, --verbose`: Enable verbose output mode
- `-h, --help`: Show help information

## Shell Completion

`seme` supports generating shell completion scripts for different shells:

```bash
# Generate bash completion script
seme completion bash > ~/.bash_completion.d/seme

# Generate zsh completion script
seme completion zsh > "${fpath[1]}/_seme"

# Generate fish completion script
seme completion fish > ~/.config/fish/completions/seme.fish
``` 