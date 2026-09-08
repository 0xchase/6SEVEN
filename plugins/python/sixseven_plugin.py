import base64
import contextlib
import ipaddress
import json
import sys


def serve(algorithm):
    model = None
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if request.get("version") != 1:
                    raise ValueError("unsupported protocol version")
                method = request["method"]
                params = request["params"]
                if method == "describe":
                    result = algorithm.describe()
                elif method == "train":
                    model = algorithm.train(params["config"], params["observations"])
                    result = None
                elif method == "load":
                    model = algorithm.load(params["model_version"], base64.b64decode(params["payload"], validate=True))
                    result = None
                elif model is None:
                    raise ValueError("model is not loaded")
                elif method == "generate":
                    limit = params["limit"]
                    if type(limit) is not int or not 1 <= limit <= 65536:
                        raise ValueError("invalid generation limit")
                    result = model.generate(limit)
                    result["addresses"] = [list(ipaddress.IPv6Address(address).packed) if isinstance(address, str) else address for address in result["addresses"]]
                elif method == "feedback":
                    model.apply_feedback(params["items"])
                    result = None
                elif method == "save":
                    result = base64.b64encode(model.save()).decode("ascii")
                elif method == "close":
                    model = None
                    result = None
                else:
                    raise ValueError(f"unknown method {method}")
                response = {"id": request["id"], "result": result}
            except Exception as error:
                response = {"id": request.get("id"), "error": {"message": str(error)}}
            output.write(json.dumps(response, separators=(",", ":")) + "\n")
            output.flush()
