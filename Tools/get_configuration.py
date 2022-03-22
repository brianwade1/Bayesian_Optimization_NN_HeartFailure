"""
Reads settings from a .ini file.

Created 4 Jan 2021

@author: Brian Wade
@version: 1.0
"""


import json
from configparser import ConfigParser
import os

class ConfigurationParameters:
    def __init__(self, ini_file):

        """
        loads parameters from ini file

        Arguments: the .ini file with the parameters
        """
        # Check if ini file in path
        file_good = os.path.exists(ini_file)
        if not file_good:
            # Check if need to add config folder
            extended_path = os.path.join('Config', ini_file)
            file_good_w_folder = os.path.exists(extended_path)
            if file_good_w_folder:
                # Change ini file to include path to Config folder
                ini_file = extended_path
            else:
                raise Exception("cannot find ini file")
                
            

        self._ini_file = ini_file
        self.current_dir = os.getcwd()


        config = ConfigParser()
        config.optionxform = lambda option: option
        config.read(self._ini_file)
        self.sections = config.sections()     

        # Set attributes using the ini file keys set to their values.
        # This method will check to see if the ini string is a int or float
        for section in self.sections:
            for key in config[section]:
                item = config[section][key].split(',')
                if len(item) > 1:
                    item = [x.strip() for x in item]
                    if True in (ele == 'True' or ele == 'False' for ele in item):
                        setattr(self, key, list(map(bool, item)))
                    else:
                        try: 
                            setattr(self, key, list(map(int, item)))
                        except:
                            try:
                                setattr(self, key, list(map(float, item)))
                            except:
                                setattr(self, key, item)
                elif section == 'architecture':
                    try:
                        setattr(self, key, list(map(int, item)))
                    except:
                        setattr(self, key, bool(item))
                else:
                    item = item[0].strip()
                    if (item == 'True' or item == 'False'):
                        setattr(self, key, bool(item))
                    else:
                        try:
                            setattr(self, key, int(item))
                        except:
                            try: 
                                setattr(self, key, float(item))
                            except:
                                setattr(self, key, item)
