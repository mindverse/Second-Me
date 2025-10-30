#!/bin/bash

################################################################################
# Second-Me Automated Installation Script for Ubuntu/Debian
# Version: 1.0.0
# Description: Automated installation with Docker and local setup options
################################################################################

set -e  # Exit on error

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Installation log file
INSTALL_LOG="install-ubuntu.log"
exec > >(tee -a "$INSTALL_LOG")
exec 2>&1

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}              Second-Me Automated Installer (Ubuntu)              ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_section() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}▶ $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_step() {
    echo -e "${BOLD}→${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root or with sudo"
        print_info "The script will ask for sudo password when needed"
        exit 1
    fi
}

# Check Ubuntu/Debian version
check_ubuntu_version() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID

        if [[ "$OS" != "ubuntu" ]] && [[ "$OS" != "debian" ]]; then
            print_warning "This script is optimized for Ubuntu/Debian"
            print_info "Detected OS: $OS $VER"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_success "Detected: $OS $VER"
        fi
    fi
}

# Check system resources
check_system_resources() {
    print_step "Checking system resources..."

    # Check RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -lt 8 ]; then
        print_warning "Only ${TOTAL_RAM}GB RAM detected. Minimum 8GB recommended."
        print_info "You can still continue with 'low' data synthesis mode"
    else
        print_success "RAM: ${TOTAL_RAM}GB"
    fi

    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 20 ]; then
        print_warning "Only ${AVAILABLE_SPACE}GB disk space available. 20GB+ recommended."
    else
        print_success "Disk space: ${AVAILABLE_SPACE}GB available"
    fi

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_success "NVIDIA GPU detected: $GPU_INFO"
        HAS_GPU=true
    else
        print_info "No NVIDIA GPU detected (CPU-only mode)"
        HAS_GPU=false
    fi
}

# Ask user for installation type
ask_installation_type() {
    print_section "Installation Type Selection"

    echo -e "${BOLD}Please choose your installation method:${NC}\n"
    echo "  1) Docker Installation (Recommended - Easiest)"
    echo "     └─ All dependencies pre-installed"
    echo "     └─ Isolated environment"
    echo "     └─ Automatic GPU detection"
    echo ""
    echo "  2) Local/Native Installation"
    echo "     └─ Better performance"
    echo "     └─ More control"
    echo "     └─ Requires manual dependency management"
    echo ""
    echo "  3) Both (Docker + Local)"
    echo "     └─ Maximum flexibility"
    echo ""

    read -p "Enter your choice (1/2/3) [1]: " INSTALL_TYPE
    INSTALL_TYPE=${INSTALL_TYPE:-1}

    case $INSTALL_TYPE in
        1)
            INSTALL_DOCKER=true
            INSTALL_LOCAL=false
            print_success "Selected: Docker Installation"
            ;;
        2)
            INSTALL_DOCKER=false
            INSTALL_LOCAL=true
            print_success "Selected: Local Installation"
            ;;
        3)
            INSTALL_DOCKER=true
            INSTALL_LOCAL=true
            print_success "Selected: Both Docker and Local"
            ;;
        *)
            print_error "Invalid choice. Defaulting to Docker."
            INSTALL_DOCKER=true
            INSTALL_LOCAL=false
            ;;
    esac
}

# Install system dependencies
install_system_dependencies() {
    print_section "Installing System Dependencies"

    print_step "Updating package lists..."
    sudo apt-get update -qq

    print_step "Installing base dependencies..."
    sudo apt-get install -y \
        curl \
        wget \
        git \
        build-essential \
        cmake \
        sqlite3 \
        libsqlite3-dev \
        ca-certificates \
        gnupg \
        lsb-release \
        unzip \
        software-properties-common

    print_success "System dependencies installed"
}

# Install Docker
install_docker() {
    print_section "Installing Docker"

    if command -v docker &> /dev/null; then
        print_info "Docker is already installed"
        docker --version
        return
    fi

    print_step "Adding Docker repository..."

    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Set up the repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    print_step "Installing Docker Engine..."
    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    print_step "Adding user to docker group..."
    sudo usermod -aG docker $USER

    print_success "Docker installed successfully"
    print_warning "You may need to log out and back in for docker group changes to take effect"
}

# Install Python
install_python() {
    print_section "Installing Python 3.12+"

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
            print_success "Python $PYTHON_VERSION already installed"
            return
        else
            print_warning "Python $PYTHON_VERSION found, but 3.12+ required"
        fi
    fi

    print_step "Adding deadsnakes PPA for Python 3.12..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq

    print_step "Installing Python 3.12..."
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip

    # Set Python 3.12 as default python3
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

    print_success "Python 3.12 installed"
    python3 --version
}

# Install Node.js
install_nodejs() {
    print_section "Installing Node.js"

    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION already installed"
        return
    fi

    print_step "Installing Node.js 20.x LTS..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs

    print_success "Node.js installed"
    node --version
    npm --version
}

# Install Poetry
install_poetry() {
    print_section "Installing Poetry"

    if command -v poetry &> /dev/null; then
        print_success "Poetry already installed"
        poetry --version
        return
    fi

    print_step "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -

    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    # Add to shell RC files
    for rc_file in ~/.bashrc ~/.zshrc; do
        if [ -f "$rc_file" ]; then
            if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$rc_file"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc_file"
            fi
        fi
    done

    print_success "Poetry installed"
    poetry --version
}

# Setup local installation
setup_local_installation() {
    print_section "Setting Up Local Installation"

    print_step "Running setup script..."
    bash ./scripts/setup.sh

    print_success "Local setup completed"
}

# Configure environment
configure_environment() {
    print_section "Configuring Environment"

    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            print_step "Creating .env file from template..."
            cp .env.example .env
            print_success ".env file created"
        else
            print_warning ".env.example not found, skipping .env creation"
        fi
    else
        print_info ".env file already exists"
    fi

    # Ask for optimization settings
    if [ "$TOTAL_RAM" -lt 16 ]; then
        print_step "Optimizing for low memory system..."
        if [ -f .env ]; then
            # Add or update settings for low memory
            if ! grep -q "DATA_SYNTHESIS_MODE" .env; then
                echo "DATA_SYNTHESIS_MODE=low" >> .env
            fi
            if ! grep -q "CONCURRENCY_THREADS" .env; then
                echo "CONCURRENCY_THREADS=1" >> .env
            fi
            print_success "Low memory optimizations applied"
        fi
    fi
}

# Setup Docker installation
setup_docker_installation() {
    print_section "Setting Up Docker Installation"

    # Check for GPU and ask user
    if [ "$HAS_GPU" = true ]; then
        echo ""
        read -p "Enable GPU support (CUDA) in Docker? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [ -z "$REPLY" ]; then
            print_step "Configuring GPU support..."

            # Install NVIDIA Container Toolkit if not present
            if ! command -v nvidia-container-toolkit &> /dev/null; then
                print_step "Installing NVIDIA Container Toolkit..."

                distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

                sudo apt-get update -qq
                sudo apt-get install -y nvidia-container-toolkit
                sudo systemctl restart docker

                print_success "NVIDIA Container Toolkit installed"
            fi

            # Mark GPU preference
            echo "gpu" > .gpu_selected
            print_success "GPU support enabled"
        fi
    fi

    print_step "Building and starting Docker containers..."
    print_info "This may take 10-30 minutes depending on your internet speed..."

    make docker-up

    print_success "Docker containers are running"
}

# Create startup scripts
create_startup_scripts() {
    print_section "Creating Convenience Scripts"

    # Create start script
    cat > start-second-me.sh << 'EOF'
#!/bin/bash
# Second-Me Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Second-Me..."

if [ -f ".docker_installed" ]; then
    echo "Starting Docker containers..."
    make docker-up
    echo ""
    echo "✓ Second-Me is running!"
    echo "  Web Interface: http://localhost:3000"
    echo "  Backend API:   http://localhost:8002"
    echo ""
    echo "To stop: make docker-down"
elif [ -f ".local_installed" ]; then
    echo "Starting local installation..."
    make start
    echo ""
    echo "✓ Second-Me is running!"
    echo "  Web Interface: http://localhost:3000"
    echo "  Backend API:   http://localhost:8002"
    echo ""
    echo "To stop: make stop"
else
    echo "Error: Installation not found"
    exit 1
fi
EOF
    chmod +x start-second-me.sh

    # Create stop script
    cat > stop-second-me.sh << 'EOF'
#!/bin/bash
# Second-Me Stop Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping Second-Me..."

if [ -f ".docker_installed" ]; then
    make docker-down
    echo "✓ Docker containers stopped"
elif [ -f ".local_installed" ]; then
    make stop
    echo "✓ Local services stopped"
fi
EOF
    chmod +x stop-second-me.sh

    print_success "Startup scripts created (start-second-me.sh, stop-second-me.sh)"
}

# Print final instructions
print_final_instructions() {
    print_section "Installation Complete!"

    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${BOLD}                  Installation Successful!                        ${NC}${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

    echo -e "${BOLD}Next Steps:${NC}\n"

    if [ "$INSTALL_DOCKER" = true ]; then
        echo -e "${CYAN}Docker Installation:${NC}"
        echo "  • Access Second-Me at: ${BOLD}http://localhost:3000${NC}"
        echo "  • Backend API at: ${BOLD}http://localhost:8002${NC}"
        echo ""
        echo "  Start: ${BOLD}./start-second-me.sh${NC} or ${BOLD}make docker-up${NC}"
        echo "  Stop:  ${BOLD}./stop-second-me.sh${NC} or ${BOLD}make docker-down${NC}"
        echo "  Logs:  ${BOLD}docker logs -f second-me-backend${NC}"
        echo ""
    fi

    if [ "$INSTALL_LOCAL" = true ]; then
        echo -e "${CYAN}Local Installation:${NC}"
        echo "  • Access Second-Me at: ${BOLD}http://localhost:3000${NC}"
        echo "  • Backend API at: ${BOLD}http://localhost:8002${NC}"
        echo ""
        echo "  Start: ${BOLD}./start-second-me.sh${NC} or ${BOLD}make start${NC}"
        echo "  Stop:  ${BOLD}./stop-second-me.sh${NC} or ${BOLD}make stop${NC}"
        echo "  Logs:  ${BOLD}tail -f logs/train/train.log${NC}"
        echo ""
    fi

    echo -e "${YELLOW}Important Notes:${NC}"
    echo "  • First startup may take a few minutes"
    echo "  • Training process can take 1-6 hours depending on data"
    echo "  • Start with small documents (1-2 PDFs) for testing"
    echo "  • Check logs if something seems stuck"

    if [ "$INSTALL_DOCKER" = true ] && [ "$INSTALL_LOCAL" = false ]; then
        echo ""
        print_warning "You may need to log out and back in for Docker permissions"
        echo "  Or run: ${BOLD}newgrp docker${NC}"
    fi

    echo ""
    echo -e "${BOLD}Useful Commands:${NC}"
    echo "  • View all commands:    ${BOLD}make help${NC}"
    echo "  • Check service status: ${BOLD}make status${NC}"
    echo "  • Restart services:     ${BOLD}make restart${NC}"

    echo ""
    echo -e "${CYAN}Documentation:${NC}"
    echo "  • Homepage: https://home.second.me/"
    echo "  • GitBook:  https://secondme.gitbook.io/secondme/"
    echo "  • Discord:  https://discord.gg/GpWHQNUwrg"

    echo ""
    echo -e "${GREEN}Installation log saved to: ${BOLD}$INSTALL_LOG${NC}\n"
}

# Error handler
error_handler() {
    print_error "An error occurred during installation"
    print_info "Check the log file for details: $INSTALL_LOG"
    print_info "You can also ask for help on Discord: https://discord.gg/GpWHQNUwrg"
    exit 1
}

trap error_handler ERR

################################################################################
# Main Installation Flow
################################################################################

main() {
    print_header

    print_info "Installation log: $INSTALL_LOG"
    print_info "Started at: $(date)"

    # Pre-flight checks
    check_root
    check_ubuntu_version
    check_system_resources

    # Ask user preferences
    ask_installation_type

    # Install system dependencies
    install_system_dependencies

    # Install Docker if requested
    if [ "$INSTALL_DOCKER" = true ]; then
        install_docker
    fi

    # Install local dependencies if requested
    if [ "$INSTALL_LOCAL" = true ]; then
        install_python
        install_nodejs
        install_poetry
    fi

    # Configure environment
    configure_environment

    # Setup installations
    if [ "$INSTALL_LOCAL" = true ]; then
        setup_local_installation
        touch .local_installed
    fi

    if [ "$INSTALL_DOCKER" = true ]; then
        setup_docker_installation
        touch .docker_installed
    fi

    # Create convenience scripts
    create_startup_scripts

    # Print final instructions
    print_final_instructions

    print_info "Installation completed at: $(date)"
}

# Run main installation
main "$@"
