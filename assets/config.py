from configparser import ConfigParser
config = ConfigParser()

config.read('config.ini')
config.add_section('test')
config.set('test', 'epochs', "100")

with open('config.ini', 'w') as f:
    config.write(f)