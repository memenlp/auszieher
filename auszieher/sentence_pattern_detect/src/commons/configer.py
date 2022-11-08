# -*- coding: utf-8 -*-
"""
Author: weisi.wu
Date: 2022/5/26
Note: Get config info from config.yaml, environment variables.
Version: v1.0
"""

import os
from yaml import load, Loader

# first, get config from config.yaml
with open(os.path.join(os.path.dirname(__file__), '../../', "./config/config.yaml"), encoding="utf-8") as fp:
    config_set = load(fp, Loader=Loader)

# second, the config depends on your environments variable ENV_PREFIX
# its default is local, and you can change it to be dev/online through environments variable command
env_name = os.getenv("ENV_PREFIX", "local")

config = config_set[env_name]
