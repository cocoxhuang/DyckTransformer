import logging
import os
from datetime import datetime


class Logger:    
    def __init__(self, cache_dir="cache", name="DyckTransformer",resume_from=None):
        """
        Initialize the logger.
        
        Args:
            cache_dir (str): Directory to store log files and other cached items
            name (str): Logger name
        """
        self.name = name

        if resume_from:
            self.cache_dir = resume_from
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.cache_dir = os.path.join(cache_dir, f"sesh_{self.timestamp}") # Create a new session directory

        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging configuration."""
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create log filename with timestamp
        log_filename = os.path.join(self.cache_dir, f"training.log")
        
        # Configure logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplication
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.info("="*50)
        self.info(f"Logging initialized. Log file: {log_filename}")
    
    def get_model_path(self, model_name="model"):
        """Get model save path with timestamp in cache directory."""
        return os.path.join(self.cache_dir, f"{model_name}.pth")

    def get_config_path(self, config_name="config"):
        """Get config save path with timestamp in cache directory."""
        return os.path.join(self.cache_dir, f"{config_name}.pth")
    
    def info(self, message):
        """Log info message to both console and file."""
        print(message)
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message (only to file, not console)."""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log warning message to both console and file."""
        print(f"WARNING: {message}")
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message to both console and file."""
        print(f"ERROR: {message}")
        self.logger.error(message)
