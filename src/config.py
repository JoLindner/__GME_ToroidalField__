import configparser
import threading
import os

class ConfigHandler:
    _instance = None
    # Protect against race conditions in multi-threaded environments: not necessary here, since only read operations are performed
    _lock = threading.Lock()

    def __new__(cls, config_path="config.ini"):
        # Check if the instance already exists
        if cls._instance is None:
            # Lock to prevent race conditions during initialization (thread-safety)
            with cls._lock:
            # double-check to ensure only one initialization
                if cls._instance is None:
                    # Check if config file exists
                    if not os.path.exists(config_path):
                        raise FileNotFoundError(f"Configuration file '{config_path}' does not exist.")
                    # Initialize the instance if it does not exist
                    cls._instance = super().__new__(cls)
                    cls._instance.config = configparser.ConfigParser()
                    cls._instance.config.read(config_path)

        # Return the existing singleton instance (after the first creation)
        return cls._instance


    def get(self, section, option, default=None):
        # Retrieve string values from the config file
        try:
            return self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def getint(self, section, option, default=0):
        # Retrieve integer values from the config file
        try:
            return self.config.getint(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def getfloat(self, section, option, default=0.0):
        # Retrieve float values from the config file
        try:
            return self.config.getfloat(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def set(self, section, option, value):
        # Set values in the config file
        # If the section doesn't exist, it will be created
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, str(value))
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)


