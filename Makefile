.PHONY: install test format lint all setup setup-china start stop restart restart-backend restart-force help docker-build docker-up docker-down docker-build-backend docker-build-frontend docker-restart-backend docker-restart-backend-fast docker-restart-backend-smart docker-restart-frontend docker-restart-all docker-check-cuda docker-use-gpu docker-use-cpu

# Colors for output
COLOR_CYAN=\033[0;36m
COLOR_BOLD=\033[1m
COLOR_GRAY=\033[0;90m
COLOR_RESET=\033[0m

# Function to display ASCII art only
define display_ascii
	@printf "\n"
	@printf "$(COLOR_CYAN)"
	@printf ' ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗\n'
	@printf ' ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝\n'
	@printf ' ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  \n'
	@printf ' ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  \n'
	@printf ' ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗\n'
	@printf ' ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝\n'
	@printf "$(COLOR_RESET)\n"
	@printf "$(COLOR_BOLD)Second-Me v1.0.0$(COLOR_RESET)\n"
	@printf "$(COLOR_GRAY)%s$(COLOR_RESET)\n\n" "$$(date)"
endef



# Check for GPU flag file and set Docker Compose file accordingly
ifeq ($(wildcard .gpu_selected),)
    # No GPU flag file found, use CPU configuration
    DOCKER_COMPOSE_FILE := docker-compose.yml
else
    # GPU flag file found, use GPU configuration
    DOCKER_COMPOSE_FILE := docker-compose-gpu.yml
endif

# Detect operating system and set environment
ifeq ($(OS),Windows_NT)
    # Set Windows variables
    WINDOWS := 1
    # Set UTF-8 code page for Windows to display Unicode characters
    SET_UTF8 := $(shell chcp 65001 >nul 2>&1 || echo)
    # No need to check for Apple Silicon on Windows
    APPLE_SILICON := 0
    # Define empty color codes for Windows to avoid display issues
    COLOR_CYAN := 
    COLOR_RESET := 
    COLOR_BOLD := 
    COLOR_GRAY := 
    COLOR_GREEN := 
    COLOR_RED := 
else
    WINDOWS := 0
    # Detect Apple Silicon on non-Windows systems
    ifeq ($(shell uname -s),Darwin)
      ifeq ($(shell uname -m),arm64)
        APPLE_SILICON := 1
        # Set environment variables for Apple Silicon
        export DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.apple
        export PLATFORM=apple
      else
        APPLE_SILICON := 0
      endif
    else
      APPLE_SILICON := 0
    endif
    # Define ANSI color codes for Unix systems
    COLOR_CYAN := \033[0;36m
    COLOR_RESET := \033[0m
    COLOR_BOLD := \033[1m
    COLOR_GRAY := \033[0;90m
    COLOR_GREEN := \033[1;32m
    COLOR_RED := \033[1;31m
endif

# Default Docker Compose configuration (non-GPU)
DOCKER_COMPOSE_FILE := docker-compose.yml

# Show help message
help:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Help $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
ifeq ($(WINDOWS),1)
	@echo SECOND-ME MAKEFILE COMMANDS
	@echo ------------------------------
	@echo.
	@echo LOCAL COMMANDS:
	@echo   make setup                 - Complete installation
	@echo   make start                 - Start all services
	@echo   make stop                  - Stop all services
	@echo   make restart               - Restart all services
	@echo   make restart-backend       - Restart only backend service
	@echo   make restart-force         - Force restart and reset data
	@echo   make status                - Show status of all services
	@echo.
	@echo DOCKER COMMANDS:
	@echo   make docker-build          - Build all Docker images
	@echo   make docker-up             - Start all Docker containers
	@echo   make docker-down           - Stop all Docker containers
	@echo   make docker-build-backend  - Build only backend Docker image
	@echo   make docker-build-frontend - Build only frontend Docker image
	@echo   make docker-restart-backend - Restart only backend container
	@echo   make docker-restart-backend-fast - Restart backend+cuda without rebuilding llama.cpp
	@echo   make docker-restart-frontend - Restart only frontend container
	@echo   make docker-restart-all    - Restart all Docker containers
	@echo   make docker-check-cuda     - Check CUDA support in containers
	@echo   make docker-use-gpu        - Switch to GPU configuration
	@echo   make docker-use-cpu        - Switch to CPU-only configuration
	@echo.
	@echo All Available Commands:
	@echo   make help                  - Show this help message
	@echo   make install               - Install project dependencies
	@echo   make test                  - Run tests
	@echo   make format                - Format code
	@echo   make lint                  - Check code style
	@echo   make all                   - Run format, lint and test
else
	@echo "$(COLOR_CYAN)"
	@echo ""
	@echo "$(COLOR_GREEN)▶ LOCAL COMMANDS:$(COLOR_RESET)"
	@echo "  make setup                 - Complete installation (global mirrors)"
	@echo "  make setup-china           - Complete installation (China mirrors)"
	@echo "  make start                 - Start all services"
	@echo "  make stop                  - Stop all services"
	@echo "  make restart               - Restart all services"
	@echo "  make restart-backend       - Restart only backend service"
	@echo "  make restart-force         - Force restart and reset data"
	@echo "  make status                - Show status of all services"
	@echo ""
	@echo "$(COLOR_GREEN)▶ DOCKER COMMANDS:$(COLOR_RESET)"
	@echo "  make docker-build          - Build all Docker images"
	@echo "  make docker-up             - Start all Docker containers"
	@echo "  make docker-down           - Stop all Docker containers"
	@echo "  make docker-build-backend  - Build only backend Docker image"
	@echo "  make docker-build-frontend - Build only frontend Docker image"
	@echo "  make docker-restart-backend - Restart only backend container"
	@echo "  make docker-restart-backend-fast - Restart backend+cuda without rebuilding llama.cpp"
	@echo "  make docker-restart-frontend - Restart only frontend container"
	@echo "  make docker-restart-all    - Restart all Docker containers"
	@echo "  make docker-check-cuda     - Check CUDA support in containers"
	@echo "  make docker-use-gpu        - Switch to GPU configuration"
	@echo "  make docker-use-cpu        - Switch to CPU-only configuration"
	@echo ""
	@echo "$(COLOR_BOLD)All Available Commands:$(COLOR_RESET)"
	@echo "  make help                  - Show this help message"
	@echo "  make install               - Install project dependencies"
	@echo "  make test                  - Run tests"
	@echo "  make format                - Format code"
	@echo "  make lint                  - Check code style"
	@echo "  make all                   - Run format, lint and test"
	@if [ "$(APPLE_SILICON)" = "1" ]; then \
		echo ""; \
		echo "$(COLOR_GREEN)▶ PLATFORM INFORMATION:$(COLOR_RESET)"; \
		echo "  Apple Silicon detected - Docker commands will use PLATFORM=apple"; \
	fi
endif

docker-use-gpu:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Use GPU $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@echo "Switching to GPU configuration..."
ifeq ($(WINDOWS),1)
	@echo GPU mode enabled. Docker commands will use docker-compose-gpu.yml
	@echo gpu > .gpu_selected
else
	@echo "$(COLOR_GREEN)GPU mode enabled. Docker commands will use docker-compose-gpu.yml$(COLOR_RESET)"
	@echo "gpu" > .gpu_selected
endif

docker-use-cpu:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Use CPU $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@echo "Switching to CPU-only configuration..."
ifeq ($(WINDOWS),1)
	@echo CPU-only mode enabled. Docker commands will use docker-compose.yml
	@rm -f .gpu_selected
else
	@echo "$(COLOR_GREEN)CPU-only mode enabled. Docker commands will use docker-compose.yml$(COLOR_RESET)"
	@rm -f .gpu_selected
endif

setup:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Setup $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/setup.sh

setup-cn:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Setup (China) $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/setup.sh --china

start:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Start $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/start.sh

stop:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Stop $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/stop.sh

restart:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Restart $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/restart.sh

restart-backend:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Restart Backend $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/restart-backend.sh

restart-force:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Restart Force $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/restart-force.sh

status:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Status $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@./scripts/status.sh

# Docker commands
# Set Docker environment variable for all Docker commands
docker-%: export IN_DOCKER_ENV=1

ifeq ($(OS),Windows_NT)
DOCKER_COMPOSE_CMD := docker compose
else
DOCKER_COMPOSE_CMD := $(shell if command -v docker-compose >/dev/null 2>&1; then echo "docker-compose"; else echo "docker compose"; fi)
endif

docker-build:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Build $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
ifeq ($(WINDOWS),1)
	@echo "Prompting for CUDA preference..."
	@scripts\prompt_cuda.bat
else
	@echo "Prompting for CUDA preference..."
	@chmod +x ./scripts/prompt_cuda.sh
	@./scripts/prompt_cuda.sh
endif
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) build

docker-up:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Up $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@echo "Building and starting Docker containers..."
ifeq ($(WINDOWS),1)
	@echo "Prompting for CUDA preference..."
	@scripts\prompt_cuda.bat
	@echo "Checking CUDA preference..."
	@cmd /c "if exist .gpu_selected ( echo CUDA support detected, using GPU configuration... & docker compose -f docker-compose-gpu.yml build & docker compose -f docker-compose-gpu.yml up -d ) else ( echo No CUDA support selected, using CPU-only configuration... & docker compose -f docker-compose.yml build & docker compose -f docker-compose.yml up -d )"
else
	@echo "Prompting for CUDA preference..."
	@chmod +x ./scripts/prompt_cuda.sh
	@./scripts/prompt_cuda.sh
	@echo "Checking CUDA preference..."
	@if [ -f .gpu_selected ]; then \
		echo "CUDA support detected, using GPU configuration..."; \
		$(DOCKER_COMPOSE_CMD) -f docker-compose-gpu.yml build; \
		$(DOCKER_COMPOSE_CMD) -f docker-compose-gpu.yml up -d; \
	else \
		echo "No CUDA support selected, using CPU-only configuration..."; \
		$(DOCKER_COMPOSE_CMD) -f docker-compose.yml build; \
		$(DOCKER_COMPOSE_CMD) -f docker-compose.yml up -d; \
	fi
endif
	@echo "Container startup complete"
	@echo "Check CUDA support with: make docker-check-cuda"

docker-down:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Down $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) down

docker-build-backend:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Build Backend $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) build backend

docker-build-frontend:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Build Frontend $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) build frontend

# Standard backend restart with complete rebuild
docker-restart-backend:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Restart Backend $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) stop backend
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) rm -f backend
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) build backend || { echo "$(COLOR_RED)❌ Backend build failed! Aborting operation...$(COLOR_RESET)"; exit 1; }
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) up -d backend


# Fast backend restart: preserves llama.cpp build
docker-restart-backend-fast:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Restart Backend (Fast) $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@echo "Smart restarting backend container (preserving llama.cpp build)..."
	@echo "Stopping backend container..."
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) stop backend
	@echo "Removing backend container..."
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) rm -f backend
	@echo "Building backend image with build-arg to skip llama.cpp build..."
ifeq ($(wildcard .gpu_selected),)
	@echo "Using CPU configuration (docker-compose.yml)..."
else
	@echo "Using GPU configuration (docker-compose-gpu.yml)..."
endif
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) build --build-arg SKIP_LLAMA_BUILD=true backend || { echo "$(COLOR_RED)❌ Backend build failed! Aborting operation...$(COLOR_RESET)"; exit 1; }
	@echo "Starting backend container..."
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) up -d backend
	@echo "Backend container smart-restarted successfully"
	@echo "Check CUDA support with: make docker-check-cuda"

docker-restart-frontend:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Restart Frontend $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) stop frontend
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) rm -f frontend
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) build frontend || { echo "$(COLOR_RED)❌ Frontend build failed! Aborting operation...$(COLOR_RESET)"; exit 1; }
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) up -d frontend

docker-restart-all:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Restart All $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) stop
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) rm -f
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) build || { echo "$(COLOR_RED)❌ Build failed! Aborting operation...$(COLOR_RESET)"; exit 1; }
	@$(DOCKER_COMPOSE_CMD) -f $(DOCKER_COMPOSE_FILE) up -d

# New command to check CUDA support in containers
docker-check-cuda:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Docker Check CUDA $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@echo "Checking CUDA support in Docker containers..."
ifeq ($(WINDOWS),1)
	@echo Running CUDA support check in backend container
	@docker exec second-me-backend /app/check_gpu_support.sh || echo No GPU support detected in backend container
else
	@echo "$(COLOR_CYAN)Running CUDA support check in backend container:$(COLOR_RESET)"
	@docker exec second-me-backend /app/check_gpu_support.sh || echo "$(COLOR_RED)No GPU support detected in backend container$(COLOR_RESET)"
endif

install:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Install $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@poetry install

test:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Test $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@poetry run pytest tests

format:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Format $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@poetry run ruff format lpm_kernel/

lint:
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)Lint $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@poetry run ruff check lpm_kernel/

all: format lint test
	@$(call display_ascii)
	@printf "\n$(COLOR_CYAN)==================== [ $(COLOR_BOLD)All $(COLOR_CYAN)] ====================$(COLOR_RESET)\n\n"
	@echo "All tasks completed!"