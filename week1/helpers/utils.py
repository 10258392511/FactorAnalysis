import tushare as ts
import yaml
import os

ROOT = os.path.abspath(__file__)
for _ in range(3):
    ROOT = os.path.dirname(ROOT)


def load_pro(env_filename=None):
    if env_filename is None:
        env_filename = os.path.join(ROOT, "configs/environ.yml")
    with open(env_filename, "r") as rf:
        token = yaml.safe_load(rf)["token"]

    pro = ts.pro_api(token)

    return pro
