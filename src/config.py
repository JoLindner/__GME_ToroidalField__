import configparser

class ConfigHandler:
    def __init__(self, config_path="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)


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


