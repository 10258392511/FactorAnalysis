import qlib

from qlib.config import REG_CN
from qlib.utils import exists_qlib_data
from qlib.tests.data import GetData
from qlib.contrib.data.handler import Alpha158


if __name__ == "__main__":
    provider_url = r"C:\Users\lenovo\.qlib\qlib_data\cn_data"
    if not exists_qlib_data(provider_url):
        GetData().qlib_data(target_dir=provider_url, region=REG_CN)
    qlib.init(provider_uri=provider_url, region=REG_CN)
    data_handler_config = {
        "start_time": "2020-01-01",
        "end_time": "2020-01-15",
        "instruments": "csi300",
    }

    data_handler = Alpha158(**data_handler_config)
    cols = data_handler.fetch()
    print(cols)
