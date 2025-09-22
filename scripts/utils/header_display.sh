#!/bin/bash

# Header display utility for Second-Me scripts
# This file contains the ASCII art header display function

# Source logging.sh for color definitions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/logging.sh"

# Version (maintain here for display purposes)
VERSION="1.0.0"

# Display title and logo
display_header() {
    local title="$1"
    
    echo ""
    echo -e "${CYAN}"
    echo ' ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗       ███╗   ███╗███████╗'
    echo ' ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗      ████╗ ████║██╔════╝'
    echo ' ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗██╔████╔██║█████╗  '
    echo ' ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝  '
    echo ' ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝      ██║ ╚═╝ ██║███████╗'
    echo ' ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝       ╚═╝     ╚═╝╚══════╝'
    echo -e "${NC}"
    echo -e "${BOLD}Second-Me Setup Script v${VERSION}${NC}"
    echo -e "${GRAY}$(date)${NC}\\n"
    
    if [ -n "$title" ]; then
        echo -e "${CYAN}====== $title ======${NC}"
        echo ""
    fi
}
