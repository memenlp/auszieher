#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：server.py
#   创 建 者：YuLianghua
#   创建日期：2022年05月20日
#   描    述：
#
#================================================================

import time
import json
import argparse
import traceback
from sanic import Sanic
from sanic import response

from auszieher.utils.logger import logger
from auszieher import Executor

parser = argparse.ArgumentParser()
parser.add_argument("--corefhosts", help="coref parser grpc hosts", \
                    default="localhost:5010")

args = parser.parse_args()

config = {
    "neuralcoref_hosts":args.corefhosts,
    "sentence_pattern" : 
        {
            "interrogative":"./data/model/sentence_pattern/interrogative/xgb.model"
        }
    }

app = Sanic('auszieher')
executor = Executor(config=config)

@app.route('/extract', methods=['POST'])
def extract(request):
    resp = {"results":[], 
            "mesg":'', 
            "success":True,
            "costtime":""}

    start = time.time()
    request_content = request.json
    query = request_content.get("query", '').strip()
    if not query:
        resp['mesg'] = "query is empty!!!"
        costtime = round((time.time()-start)*1000, 3)
        resp['costtime'] = f"{costtime}ms"
        return response.json(resp)
    try:
        results = executor.extract(query, coref=True)
    except Exception as e:
        mesg = traceback.format_exc()
        logger.error(f"extract failed: {mesg}")
        resp["success"] = False
        resp["mesg"] = mesg
        costtime = round((time.time()-start)*1000, 3)
        resp['costtime'] = f"{costtime}ms"
        return response.json(resp)

    # -debug info
    for result in results:
        logger.debug(f"{result}")
        resp["results"].append(result.to_dict())

    costtime = round((time.time()-start)*1000, 3)
    resp["costtime"] = f"{costtime}ms"
    r = json.dumps(resp, ensure_ascii=False)

    return response.text(r)

if __name__ == '__main__':
    try:
        logger.info("Auszieher extract server running...")
        app.run(host='0.0.0.0', port=5555)
    except Exception as e:
        logger.error(f"server error!!! : {traceback.format_exc()} ")
