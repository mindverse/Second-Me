import { useEffect, useRef, useState } from 'react';

interface TrainingLogProps {
  trainingDetails: {
    message: string;
    timestamp: string;
  }[];
}

const TrainingLog: React.FC<TrainingLogProps> = ({ trainingDetails }: TrainingLogProps) => {
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const userScrollTimeout = useRef<NodeJS.Timeout | null>(null);
  const [isAutoScrollEnabled, setIsAutoScrollEnabled] = useState(true);

  // Function to clean ANSI escape sequences from text
  const cleanAnsiSequences = (text: string): string => {
    // More comprehensive ANSI escape sequence regex pattern
    // This handles various ANSI escape sequences including:
    // - Cursor movement: \x1b[nA, \x1b[nB, \x1b[nC, \x1b[nD
    // - Clear screen: \x1b[K, \x1b[J
    // - Color codes: \x1b[38;5;n, \x1b[48;5;n, \x1b[0m, etc.
    // - Progress bar characters: █, ▓, ▒, ░, ▏, ▕
    const ansiEscape = /\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g;
    
    // Remove ANSI escape sequences
    let cleaned = text.replace(ansiEscape, '');
    
    // Also remove common progress bar characters that might cause display issues
    const progressChars = /[█▓▒░▏▕]/g;
    cleaned = cleaned.replace(progressChars, '=');
    
    return cleaned;
  };

  // Smooth scroll console to bottom
  const smoothScrollConsole = () => {
    if (consoleEndRef.current && !isUserScrolling) {
      const consoleContainer = consoleEndRef.current;

      if (consoleContainer instanceof HTMLElement) {
        consoleContainer.scrollTo({
          top: consoleContainer.scrollHeight,
          behavior: 'smooth'
        });
      }
    }
  };

  useEffect(() => {
    // Set up scroll event listener to detect user scrolling
    const handleUserScroll = () => {
      if (!consoleEndRef.current) return;
      
      const consoleContainer = consoleEndRef.current.closest('.overflow-y-auto');
      
      if (!(consoleContainer instanceof HTMLElement)) return;
      
      // Check if scrolled away from bottom
      const isScrolledToBottom = 
        Math.abs((consoleContainer.scrollHeight - consoleContainer.scrollTop) - consoleContainer.clientHeight) < 50;
      
      // If scrolled away from bottom, consider it manual scrolling
      if (!isScrolledToBottom) {
        setIsUserScrolling(true);

        // Clear any existing timeout
        if (userScrollTimeout.current) {
          clearTimeout(userScrollTimeout.current);
        }

        // Reset the flag after a delay
        userScrollTimeout.current = setTimeout(() => {
          setIsUserScrolling(false);
        }, 5000); // 5 seconds delay before allowing auto-scroll again
      } else {
        // If at bottom, not considered manual scrolling
        setIsUserScrolling(false);
        if (userScrollTimeout.current) {
          clearTimeout(userScrollTimeout.current);
          userScrollTimeout.current = null;
        }
      }
    };

    // Find the console container and attach the scroll listener
    if (consoleEndRef.current) {
      const consoleContainer = consoleEndRef.current;

      if (consoleContainer instanceof HTMLElement) {
        consoleContainer.addEventListener('scroll', handleUserScroll);

        // Cleanup function
        return () => {
          consoleContainer.removeEventListener('scroll', handleUserScroll);

          if (userScrollTimeout.current) {
            clearTimeout(userScrollTimeout.current);
          }
        };
      }
    }
  }, []);

  useEffect(() => {
    if (trainingDetails.length > 0) {
      smoothScrollConsole();
    }
  }, [trainingDetails, isAutoScrollEnabled]);

  const toggleAutoScroll = () => {
    setIsAutoScrollEnabled(!isAutoScrollEnabled);
    if (!isAutoScrollEnabled) {
      // If we're re-enabling auto-scroll, scroll to bottom immediately
      setIsUserScrolling(false);
      setTimeout(smoothScrollConsole, 50);
    }
  };

  return (
    <div className="mt-4">
      <h4 className="text-sm font-medium text-gray-700 mb-2">Training Log</h4>
      <div
        ref={consoleEndRef}
        className="bg-gray-900 rounded-lg p-4 h-[600px] overflow-y-auto font-mono text-xs"
      >
        <div className="space-y-1">
          {trainingDetails.length > 0 ? (
            trainingDetails.map((detail, index) => (
              <div key={detail.timestamp + detail.message + index} className="text-gray-300">
                {cleanAnsiSequences(detail.message)}
              </div>
            ))
          ) : (
            <div className="text-gray-300">
              No training logs available. Start training to see logs here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingLog;
