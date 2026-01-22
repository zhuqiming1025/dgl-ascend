#!/bin/bash
set -e  # 关键指令：遇到错误立即停止

cd /home/zqm1/dgl-ascend
bash script/build_dgl_ascend.sh
cd /home/zqm1/dgl-ascend/python
pip install -e .
python3 /home/zqm1/dgl-ascend/tests/python/test_dispatch.py