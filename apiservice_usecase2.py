from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

APIKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

# Sample data representing road conditions
road_conditions_data = {}


@app.route("/auth/getAPIKey", methods=["POST"])
def get_api_key():
    data = request.get_json()
    account = data.get("account")
    password = data.get("password")
    if account == "admin12345" or password == "admin12345":
        return jsonify({"status": "success", "APIKey": APIKey})
    else:
        return (
            jsonify(
                {
                    "status": "Unauthorized",
                    "message": "Account or password is not correct.",
                }
            ),
            401,
        )


@app.route("/model/ml_usecase_2/train", methods=["GET"])
def get_road_conditions():
    APIKey_ = request.args.get("APIKey")
    if APIKey_ == APIKey:
        return jsonify(
            {"status": "success", "jobId": datetime.today().strftime("%Y%m%d%H%M%S")}
        )
    else:
        return (
            jsonify({"status": "Unauthorized", "message": "Unauthorized API key"}),
            401,
        )


@app.route("/roadcondition/update", methods=["POST"])
def update_road_condition():
    data = request.get_json()
    APIKey_ = data.get("APIKey")
    lon = data.get("longtitude")
    lat = data.get("latitude")
    damage_type = data.get("damage_type")
    severity = data.get("severity")

    if APIKey_ == APIKey:

        id = f'wtl_{str(datetime.today().strftime("%Y%m%d%H%M%S"))}'

        road_conditions_data[id] = {
            "longtitude": lon,
            "latitude": lat,
            "damage_type": damage_type,
            "severity": severity,
            "reported_at": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return (
            jsonify(
                {
                    "status": "success",
                    "report_id": id,
                    "message": "Data updated successfully",
                }
            ),
            201,
        )
    else:
        return (
            jsonify({"status": "Unauthorized", "message": "Unauthorized API key"}),
            401,
        )


@app.route("/roadcondition/notifications", methods=["GET"])
def road_condition_notification():
    APIKey_ = request.args.get("APIKey")

    if APIKey_ == APIKey:
        return jsonify({"status": "success", "data": road_conditions_data}), 201
    else:
        return (
            jsonify({"status": "Unauthorized", "message": "Unauthorized API key"}),
            401,
        )


@app.route("/roadcondition/delete", methods=["DELETE"])
def road_condition_delete():

    APIKey_ = request.args.get("APIKey")
    report_id = request.args.get("report_id")

    if APIKey_ == APIKey:
        del road_conditions_data[report_id]
        return jsonify({"status": "success", "data": road_conditions_data}), 201
    else:
        return (
            jsonify({"status": "Unauthorized", "message": "Unauthorized API key"}),
            401,
        )


@app.route("/roadcondition/modify", methods=["PUT"])
def road_condition_modify():

    data = request.get_json()
    APIKey_ = data.get("APIKey")
    report_id = data.get("report_id")
    severity = data.get("severity")

    if APIKey_ == APIKey:
        road_conditions_data[report_id]["severity"] = severity
        return jsonify({"status": "success", "data": road_conditions_data}), 201
    else:
        return (
            jsonify({"status": "Unauthorized", "message": "Unauthorized API key"}),
            401,
        )


if __name__ == "__main__":
    app.run(debug=True)
