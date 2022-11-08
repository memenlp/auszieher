# -*- coding: utf-8 -*-
"""
Author: weisi.wu
Date: 2022/5/26
Note: response methods
Version: v1.0
"""

import traceback
from enum import Enum

import sanic


class ResponseCode(Enum):
    OK = True
    FAIL = False


def response_json(code=ResponseCode.OK, message="", status=200, **data):
    """
    The data format for response
    :param code: response code
    :param message: error message
    :param status: the status code
    :param data: customized response data
    :return:
    """
    return sanic.response.json({
        "success": code.value,
        "error_msg": message,
        **data
    }, status, headers={"Access-Control-Allow-Origin": "*"})


def handle_exception(request, e):
    """
    if exception occurs during process, this method
    will capture the exception before responding to client.
    """
    traceback.print_exc()

    code = ResponseCode.FAIL
    message = repr(e)
    status = 500

    data = {}
    if request.app.config.get("DEBUG"):
        data["exception"] = traceback.format_exc()

    return response_json(code, message, status, **data)

