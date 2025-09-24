"""
WeChat Bot with Second-Me Integration
-------------------------------------
This bot integrates with wxpy and Second-Me to process and respond
to WeChat messages intelligently.
"""

import json
import logging
import os
import sys
from typing import Any, Optional

from dotenv import load_dotenv
from wxpy import Bot, Message

# Add Second-Me path dynamically
sys.path.append(os.path.join(os.path.dirname(__file__), "lpm_kernel"))

from lpm_kernel.kernel import SecondMeKernel  # noqa: E402
from lpm_kernel.utils import load_config  # noqa: E402

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("WeChatBot")

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()


class WeChatBot:
    """A WeChat Bot that integrates with Second-Me AI kernel."""

    def __init__(self, cache_path: str = "wxpy.pkl") -> None:
        """Initialize WeChatBot with wxpy and Second-Me."""
        try:
            self.bot: Bot = Bot(cache_path=cache_path, console_qr=False)
            logger.info("✅ WeChat bot initialized successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize WeChat bot: {e}")
            raise

        # Initialize Second-Me
        self.second_me: Optional[SecondMeKernel] = None
        try:
            config = load_config()
            self.second_me = SecondMeKernel(config)
            logger.info("✅ Second-Me initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Second-Me: {e}")

    def handle_message(self, msg: Message) -> None:
        """
        Handle incoming WeChat messages and respond using Second-Me.
        """
        try:
            content: str = msg.text or ""
            sender: str = getattr(msg.sender, "name", "Unknown")

            logger.info(f"📩 Message received from {sender}: {content}")

            if not self.second_me:
                msg.reply("⚠️ Sorry, Second-Me service is not available right now.")
                return

            response: Any = self.second_me.process_message(content)

            # Convert dict responses to JSON string
            if isinstance(response, dict):
                response = json.dumps(response, ensure_ascii=False, indent=2)

            msg.reply(str(response))

        except Exception as e:
            logger.exception(f"❌ Error handling message: {e}")
            msg.reply("⚠️ An unexpected error occurred while processing your message.")

    def run(self) -> None:
        """
        Run the WeChat bot and keep it alive.
        """
        try:
            @self.bot.register()
            def _(msg: Message) -> None:  # noqa: F811
                self.handle_message(msg)

            logger.info("🚀 WeChat bot is now running. Press Ctrl+C to stop.")
            self.bot.join()

        except KeyboardInterrupt:
            logger.warning("🛑 Bot stopped manually by user.")
            self.bot.logout()
        except Exception as e:
            logger.exception(f"❌ Fatal error while running the bot: {e}")
            self.bot.logout()


if __name__ == "__main__":
    try:
        bot = WeChatBot()
        bot.run()
    except Exception as e:
        logger.critical(f"❌ Unable to start bot: {e}")
        sys.exit(1)
