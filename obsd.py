import logging
import uvicorn
from fastapi import FastAPI
import requests
import numpy as np
import os
from os.path import join, dirname, abspath
from torch import nn
import torch
import calendar
import holidays
import itertools
from datetime import datetime, timedelta
import pickle
import pandas as pd
import json as J

from ..config import (
    RESOLUTION, LOCALHOST, COORDINATOR_PORT, POOLING_PORT, OPENEMS_PORT,
    OBSD_PORT, OBSD_LOG, HISTORY_HORIZON_IN_HOURS, PREDICTION_HORIZON_IN_HOURS,
    SIMULATION_REAL_TIME_FACTOR
)
from ..utils.utils import (
    action_handler, interface_handler, get_simulation_time, jsonizable, if_unit_test, if_server,
    make_log_str, log_and_print, infer_cur_interval_start, infer_historical_interval_start,
    infer_last_interval_start, LogConfig, time_2_datetime, if_real, START_TIME, INTERVAL_IN_SEC,
    time_2_sec, sec_2_datetime, datetime_2_time
)
from ..db.db import insert_metric_db, insert_gui_db, connect_measurement_db
from ..interface.interface_class.commons_schema import (
    SimulatorInfo, SuccessResponse, DeviceStatusSeries, Order, ScheduleSery, P2PClearedOrder, DeviceMeasurementSeries
)
from ..interface.interface_class.obsz2obsd_schema import Obsz2Obsd
from ..interface.interface_class.obsd2obsz_schema import Obsd2Obsz
from ..interface.interface_class.obsd2pooling_schema import Obsd2Pooling
from ..interface.interface_class.obsd2openems_schema import Obsd2Openems
from ..interface.interface_class.obsd2bkv_schema import Obsd2Bkv
from ..interface.interface_class.agent2smbs_schema import Agent2Smbs
from ..interface.interface_class.pooling2agent_schema import Pooling2Agent
from ..interface.interface_class.smbs2obsd_schema import Smbs2Obsd
from ..interface.interface_class.openems2obsd_schema import Openems2Obsd
from ..interface.interface_class.bws2obsd_schema import Bws2Obsd

LogConfig.init()
LOGGER = logging.getLogger(__name__)

obsd_client = FastAPI()
parameter_dict = {}
if if_unit_test():
    from ..community_config.community_config_20 import communityConfigData

    user_idx = int(os.getenv("USER_IDX"))
    user_config = communityConfigData["prosumers"][user_idx]
    prosumerId = user_config["prosumerId"]
    device_ids = [device["deviceId"] for device in user_config["devices"]]
    device_types = [device["deviceType"] for device in user_config["devices"]]
else:
    with open('./config/prosumer_config.json', 'r', encoding='utf-8') as file:
        conf = J.load(file)
    user_config = conf['prosumer']
    prosumerId = user_config["prosumerId"]
    device_ids = [device["deviceId"] for device in user_config["devices"]]
    device_types = [device["deviceType"] for device in user_config["devices"]]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * 2)
        self.fc2 = nn.Linear(in_dim * 2, in_dim * 2)
        self.fc3 = nn.Linear(in_dim * 2, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x)) * 2 - 1.
        return x


def get(simulator_info: SimulatorInfo, pre_jsons=None, post_jsons=None):
    msg_json = None
    if simulator_info.action == "individual_power_prediction":
        if simulator_info.interface is None:
            msg_json = jsonizable(SuccessResponse(msg="successful!").dict())
        elif simulator_info.interface == "openems2obsd" or simulator_info.interface == "obsz2obsd":
            msg_json = jsonizable(simulator_info)
        elif simulator_info.interface == "obsd2obsz":
            device_status_list = []
            for deviceStatusSeries in parameter_dict["openems2obsd"]["deviceStatusSeriesList"]:
                # Check if the device in config
                legal_device_type = deviceStatusSeries["deviceType"] in device_types
                legal_device_id = deviceStatusSeries["deviceID"] in device_ids
                if not (legal_device_type and legal_device_id):
                    LOGGER.warning(
                        f"Device {deviceStatusSeries['deviceType']} with device Id "
                        f"{deviceStatusSeries['deviceId']} received from OpenEMS "
                        f"does not exists in user config! Skip it for predcition!"
                    )
                    continue
                else:
                    device_idx = [device["deviceId"] for device in user_config["devices"]].index(deviceStatusSeries["deviceID"])

                # Skip prediction for battery
                if deviceStatusSeries["deviceType"] == "battery":
                    continue

                if deviceStatusSeries["deviceType"] in ["load", "wind", "solar"]:
                    # # Init, load MLP weights with the prosumer Id and turn it into evaluation model
                    # # Current supported energy devices: load, wind and solar
                    # if deviceStatusSeries["deviceType"] == "load" or deviceStatusSeries["deviceType"] == "solar":
                    #     feat_dim = 3 * (HISTORY_HORIZON_IN_HOURS + PREDICTION_HORIZON_IN_HOURS) + HISTORY_HORIZON_IN_HOURS
                    # elif deviceStatusSeries["deviceType"] == "wind":
                    #     feat_dim = 4 * (HISTORY_HORIZON_IN_HOURS + PREDICTION_HORIZON_IN_HOURS) + HISTORY_HORIZON_IN_HOURS
                    #
                    # feat_dim = feat_dim * RESOLUTION
                    # out_dim = PREDICTION_HORIZON_IN_HOURS * RESOLUTION
                    #
                    # model = MLP(in_dim=feat_dim, out_dim=out_dim)
                    # model_path = join(
                    #     dirname(dirname(abspath(__file__))),
                    #     f"resources/data/ml/model_weights/{deviceStatusSeries['deviceType']}/{prosumerId}.pt"
                    # )
                    # model.load_state_dict(torch.load(model_path))
                    # model.eval()
                    #
                    # # Prepare the input data for the prediction model depending on the energy device type
                    # datetime_objects = [
                    #     infer_historical_interval_start(simulator_info, interval_diff=-intervan_idx_)[0] for intervan_idx_ in
                    #     range(-HISTORY_HORIZON_IN_HOURS * RESOLUTION, PREDICTION_HORIZON_IN_HOURS * RESOLUTION)
                    # ]
                    # normalized_date = [2 * date.timetuple().tm_yday / (365 + calendar.isleap(date.year)) for date in datetime_objects]
                    # normalized_time = [date.hour * 60 + date.minute for date in datetime_objects]
                    # normalized_time = 2. * np.array(normalized_time) / (24 * 60) - 1
                    #
                    # if deviceStatusSeries["measuringUnit"] == "kW":
                    #     power = np.array(deviceStatusSeries["dataSeries"])
                    # elif deviceStatusSeries["measuringUnit"] == "W":
                    #     power = np.array(deviceStatusSeries["dataSeries"]) / 1000
                    # else:
                    #     raise ValueError(
                    #         f"The power unit {deviceStatusSeries['measuringUnit']} "
                    #         f"is not supported!"
                    #     )
                    # if user_config["devices"][device_idx]["peakPower"] > 0:
                    #     denominator = user_config["devices"][device_idx]["peakPower"]
                    # else:
                    #     denominator = 0
                    # normalized_power = 2. * power / denominator - 1
                    #
                    # if deviceStatusSeries["deviceType"] == 'load':
                    #     years = set([date.year for date in datetime_objects])
                    #     de_holidays = list(itertools.chain.from_iterable([holidays.Germany(years=year) for year in years]))
                    #     normalized_holidays = []
                    #     for date in datetime_objects:
                    #         if datetime.strptime(date.strftime('%Y-%m-%d'), '%Y-%m-%d') in de_holidays:
                    #             normalized_holidays.append(1)
                    #         elif date.weekday() >= 5:
                    #             normalized_holidays.append(0)
                    #         else:
                    #             normalized_holidays.append(-1)
                    #     feature = np.array(normalized_holidays).reshape(-1, 1)
                    # elif deviceStatusSeries["deviceType"] == 'wind':
                    #     for weatherSeries in parameter_dict["bws2obsd"]["weatherData"]:
                    #         if weatherSeries["weatherDataType"] == "wspd":
                    #             wspd = weatherSeries["dataSeries"]
                    #         elif weatherSeries["weatherDataType"] == "wd":
                    #             wd = weatherSeries["dataSeries"]
                    #     norm_filepath = join(
                    #         dirname(dirname(abspath(__file__))),
                    #         f"resources/data/featured_data/{deviceStatusSeries['deviceType']}/{prosumerId}.pkl"
                    #     )
                    #     with open(norm_filepath, "rb") as f:
                    #         max_wspd = pickle.load(f)["max_wspd"]
                    #     normalized_wspd = 2. * np.array(wspd) / max_wspd - 1
                    #     normalized_wd = 2. * np.array(wd) / 360 - 1
                    #     feature = np.stack([normalized_wspd, normalized_wd], axis=-1)
                    # elif deviceStatusSeries["deviceType"] == 'solar':
                    #     for weatherSeries in parameter_dict["bws2obsd"]["weatherData"]:
                    #         if weatherSeries["weatherDataType"] == "tsun":
                    #             tsun = weatherSeries["dataSeries"]
                    #     norm_filepath = join(
                    #         dirname(dirname(abspath(__file__))),
                    #         f"resources/data/featured_data/{deviceStatusSeries['deviceType']}/{prosumerId}.pkl"
                    #     )
                    #     with open(norm_filepath, "rb") as f:
                    #         max_tsun = pickle.load(f)["max_tsun"]
                    #     normalized_tsun = 2. * np.array(tsun) / max_tsun - 1
                    #     feature = normalized_tsun.reshape(-1, 1)
                    # if if_unit_test():
                    #     power_series = [0 for _ in range(PREDICTION_HORIZON_IN_HOURS)]
                    # else:
                    #     model_input = np.concatenate(
                    #         [
                    #             np.concatenate(
                    #                 [
                    #                     feature[: HISTORY_HORIZON_IN_HOURS * RESOLUTION],
                    #                     np.stack(
                    #                         [
                    #                             normalized_date[: HISTORY_HORIZON_IN_HOURS * RESOLUTION],
                    #                             normalized_time[: HISTORY_HORIZON_IN_HOURS * RESOLUTION],
                    #                             normalized_power
                    #                         ],
                    #                         axis=-1
                    #                     )
                    #                 ],
                    #                 axis=-1
                    #             ).reshape(-1),
                    #             np.concatenate(
                    #                 [
                    #                     feature[HISTORY_HORIZON_IN_HOURS * RESOLUTION:],
                    #                     np.stack(
                    #                         [
                    #                             normalized_date[HISTORY_HORIZON_IN_HOURS * RESOLUTION:],
                    #                             normalized_time[HISTORY_HORIZON_IN_HOURS * RESOLUTION:],
                    #                         ],
                    #                         axis=-1
                    #                     )
                    #                 ],
                    #                 axis=-1
                    #             ).reshape(-1)
                    #         ],
                    #         axis=-1
                    #     ).astype(np.float32)
                    #     model_input = torch.FloatTensor(model_input)
                    #     with torch.no_grad():
                    #         pred = user_config["devices"][device_idx]["peakPower"] * (model(model_input) + 1) / 2.
                    #     power_series = [round(x, 3) for x in pred.tolist()]
                    device_type_folder = join(
                        dirname(dirname(abspath(__file__))),
                        f"resources/data/featured_data/{deviceStatusSeries['deviceType']}"
                    )
                    filepath = join(device_type_folder, f"simulation/{prosumerId}.csv")

                    df = pd.read_csv(filepath)

                    seconds = (time_2_sec(datetime.now()) - simulator_info.time_info.sim_start_in_sec) * SIMULATION_REAL_TIME_FACTOR
                    current_time = START_TIME + timedelta(seconds=seconds)

                    # For data alignment, there is no date 02-29-2020
                    if current_time.month == 2 and current_time.day == 29:
                        current_time = START_TIME + timedelta(days=1)

                    row_idx = int((current_time - START_TIME).total_seconds() // INTERVAL_IN_SEC)
                    data_series = (df.iloc[row_idx: row_idx + PREDICTION_HORIZON_IN_HOURS * RESOLUTION]["normalized_power"] + 1) / 2.
                    data_series[1:] = data_series[1:] + np.random.normal(0, .03, len(data_series) - 1)
                    data_series.loc[data_series < 0] = 0
                    power_series = (user_config["devices"][device_idx]["peakPower"] * data_series).tolist()
                elif deviceStatusSeries["deviceType"] == "chp":
                    for deviceStatusSeries in parameter_dict["openems2obsd"]["deviceStatusSeriesList"]:
                        power_series = []
                        for _ in deviceStatusSeries["dataSeries"][-RESOLUTION * 24:]:
                            power_series.append(_)
                        for _ in range(PREDICTION_HORIZON_IN_HOURS * RESOLUTION):
                            if sum(power_series[-RESOLUTION * 24 + 1:]) + power_series[-1] > sum(power_series[-RESOLUTION * 24:]):
                                power_series.append(0)
                            else:
                                power_series.append(power_series[-1])
                        power_series = power_series[-PREDICTION_HORIZON_IN_HOURS * RESOLUTION:]
                device_status_list.append(
                    DeviceStatusSeries(
                        deviceID=deviceStatusSeries["deviceID"],
                        deviceType=deviceStatusSeries["deviceType"],
                        startTime=infer_cur_interval_start(simulator_info)[1],
                        stepDuration='PT15M',
                        stepCount=len(power_series),
                        measuringUnit="kW",
                        dataSeries=power_series
                    )
                )

            obsd2obsz = Obsd2Obsz(
                **{"prosumerId": prosumerId, "deviceStatusSeriesList": device_status_list}
            )
            msg_json = jsonizable(obsd2obsz.dict(exclude_unset=True, exclude_none=True))
            parameter_dict["obsd2obsz"] = msg_json

            if if_server():
                simulation_time, simulated_time = get_simulation_time(simulator_info)
                for device_series in msg_json["deviceStatusSeriesList"]:
                    insert_metric_db(
                        "prediction",
                        {
                            "simulation_time": simulation_time,
                            "simulated_time": simulated_time,
                            "prosumer_id": prosumerId,
                            "device_id": device_series["deviceID"],
                            "device_type": device_series["deviceType"],
                            "measuring_unit": device_series["measuringUnit"],
                            "start_time": sec_2_datetime(time_2_sec(datetime_2_time(device_series["startTime"])) + 3600 / RESOLUTION),
                            "step_count": device_series["stepCount"],
                            "step_duration": device_series["stepDuration"],
                            "data_series": [round(_, 3) for _ in device_series["dataSeries"]],
                        }
                    )
        elif simulator_info.interface[:8] == "bws2obsd":
            msg_json = jsonizable(simulator_info)
    elif simulator_info.action == "p2p_clearing":
        if simulator_info.interface == "obsd2pooling":
            greyEnergyKwh = 0
            greenEnergyKwh = 0
            for deviceStatusSeries in parameter_dict["obsd2obsz"]["deviceStatusSeriesList"]:
                if deviceStatusSeries["measuringUnit"] != "kW":
                    raise ValueError(f"The power unit {deviceStatusSeries['measuringUnit']} not supported!")

                if deviceStatusSeries["deviceType"] == "load":
                    greyEnergyKwh -= abs(deviceStatusSeries["dataSeries"][2]) / RESOLUTION

                if deviceStatusSeries["deviceType"] == "wind" or deviceStatusSeries["deviceType"] == "solar":
                    greenEnergyKwh += abs(deviceStatusSeries["dataSeries"][2]) / RESOLUTION

            # Get the predicted clear price
            if parameter_dict["obsz2obsd"]["predictedClearingPrice"]["measuringUnit"] != "EUR/kWh":
                raise ValueError(
                    f"The energy unit "
                    f"{parameter_dict['obsz2obsd']['predictedClearingPrice']['measuringUnit']} undefined!"
                )
            marketPriceEurocentPerKwh = parameter_dict["obsz2obsd"]["predictedClearingPrice"]["dataSeries"][2] * 100
            msg_json = jsonizable(
                Obsd2Pooling(
                    targetComponentId="PoolingPlatform",
                    prosumerId=prosumerId,
                    timeSlot=parameter_dict["obsz2obsd"]["timeSlot"],
                    # TODO: default user preference of selling or buying half of the energy demand from the pooling platform
                    greyEnergyKwh=greyEnergyKwh,
                    greenEnergyKwh=greenEnergyKwh,
                    marketPriceEurocentPerKwh=marketPriceEurocentPerKwh,
                    grid_fees=parameter_dict["obsz2obsd"]["grid_fees"],
                    grid_locations=parameter_dict["obsz2obsd"]["grid_locations"]
                )
            )
            write_db(simulator_info, "get", json=msg_json)
    elif simulator_info.action == "scheduling":
        if simulator_info.interface is None:
            msg_json = jsonizable(SuccessResponse(msg="successful!").dict())
        elif simulator_info.interface == "obsz2obsd":
            msg_json = jsonizable(simulator_info)
        elif simulator_info.interface == "smbs2obsd":
            msg_json = {
                "marketTimeSlot": infer_last_interval_start(simulator_info)[1],
                "prosumerId": prosumerId
            }
        elif simulator_info.interface == "pooling2agent":
            msg_json = jsonizable({"simulator_info": simulator_info})
        elif simulator_info.interface == "obsd2smbs" or simulator_info.interface == "agent2smbs":
            msg_json = jsonizable(
                Agent2Smbs(
                    prosumerId=prosumerId,
                    marketTimeSlot=infer_cur_interval_start(simulator_info)[1],
                    orders=[
                        Order(
                            orderType=order["order_type"],
                            energy=order["traded_energy"],
                            energyUnit="kWh",
                            priceRate=order["price_rate"],
                            priceUnit="EUR/kWh"
                        ) for order in parameter_dict["order"]
                    ],
                    p2pClearedOrders=[
                        P2PClearedOrder(
                            orderType="bid" if prosumerId == p2p_order["producerID"] else "ask",
                            poolID=p2p_order["poolID"],
                            tradePartnerID=p2p_order["producerID"] if p2p_order["producerID"] != prosumerId else p2p_order["consumerID"],
                            energy=p2p_order["energyKwh"],
                            energyUnit="kWh",
                            priceRate=p2p_order["energyPoolPrice"],
                            priceUnit="EUR/kWh"
                        ) for p2p_order in parameter_dict["pooling2obsd"]["matches"]
                    ]
                )
            )
            write_db(simulator_info, "get", json=msg_json)
            print_log_str, log_str = make_log_str(simulator_info, msg_json)
            log_and_print(OBSD_LOG, print_log_str, log_str)
        elif simulator_info.interface == "obsd2openems":
            if "battery" in device_types:
                battery = user_config["devices"][device_types.index("battery")]
                scheduleSeriesList = [
                    ScheduleSery(
                        deviceID=battery["deviceId"],
                        deviceType="battery",
                        maxGenerationPower=battery["chargePower"],
                        maxConsumptionPower=battery["chargePower"],
                        startTime=infer_cur_interval_start(simulator_info)[1],
                        stepDuration="PT15M",
                        stepCount=1,
                        dataSeries=[parameter_dict["schedule"]["charging_power_in_kW"] -parameter_dict["schedule"]["discharging_power_in_kW"]],
                        measuringUnit="kW"
                    )
                ]
            else:
                scheduleSeriesList = None
            json_ = {
                "prosumerId": prosumerId,
                "scheduleSeriesList": scheduleSeriesList
            }
            msg_json = jsonizable({"simulator_info": simulator_info, "obsd2openems": Obsd2Openems(**json_)})
    elif simulator_info.action == "send_measurement":
        if simulator_info.interface is None:
            msg_json = jsonizable(SuccessResponse(msg="successful!").dict())
        elif simulator_info.interface == "obsd2bkv":
            if if_unit_test():
                json_path = join(dirname(dirname(abspath(__file__))), "interface/json_interface/obsd2bkv.sample.json")
                with open(json_path) as f:
                    obsd2bkv = J.load(f)
            else:
                engine, connection = connect_measurement_db()
                df = pd.read_sql(f"SELECT * FROM  measurement_record", con=engine)

                device_measurement_series = []
                for device_id in device_ids:
                    selected_df = df[df["deviceId"] == device_id]
                    if selected_df.shape[0] == 0:
                        continue

                    timeStamps = selected_df["timeStamp"].tolist()
                    dataSeries = selected_df["data"].tolist()
                    timeStampedDataSeries = [f"{ts},{d}" for ts, d in zip(timeStamps, dataSeries)]
                    device_measurement_series.append(
                        DeviceMeasurementSeries(
                            deviceID=device_id,
                            deviceType=device_types[device_ids.index(device_id)],
                            measuringUnit="kW",
                            timeStampedDataSeries=timeStampedDataSeries
                        )
                    )
                engine.execute(f"DELETE FROM measurement_record WHERE \"prosumerId\" = '{prosumerId}';")
                if if_server():
                    time_slot = get_simulation_time(simulator_info)[1]
                else:
                    time_slot = time_2_datetime(datetime.now())
                if "obsd2obsz" in parameter_dict:
                    obsd2bkv = Obsd2Bkv(
                        prosumerId=prosumerId,
                        timeSlot=time_slot,
                        deviceMeasurementSeries=device_measurement_series,
                        deviceStatusSeriesList=parameter_dict["obsd2obsz"]["deviceStatusSeriesList"]
                    )
                else:
                    obsd2bkv = Obsd2Bkv(
                        prosumerId=prosumerId,
                        timeSlot=time_slot,
                        deviceMeasurementSeries=device_measurement_series
                    )
                connection.close()
            msg_json = jsonizable(obsd2bkv)

    if msg_json is None:
        raise NotImplementedError(
            f"The get function undefined at action {simulator_info.action}, interface {simulator_info.interface}!"
        )
    return msg_json


def update(simulator_info: SimulatorInfo, msg_json=None):
    if simulator_info.action == "individual_power_prediction":
        parameter_dict["openems2obsd"] = jsonizable(Openems2Obsd(**msg_json[0]))
        parameter_dict["bws2obsd"] = jsonizable(Bws2Obsd(**msg_json[1]))
        parameter_dict["obsz2obsd"] = jsonizable(Obsz2Obsd(**msg_json[2]))
        write_db(simulator_info, handler_type="update", openems2obsd=parameter_dict["openems2obsd"])
    elif simulator_info.action == "scheduling":
        parameter_dict["obsz2obsd"] = jsonizable(Obsz2Obsd(**msg_json[0]))
        parameter_dict["smbs2obsd"] = jsonizable(Smbs2Obsd(**msg_json[1]))
        parameter_dict["pooling2obsd"] = jsonizable(Pooling2Agent(**msg_json[2]))

        # Write metric db for trading
        write_db(
            simulator_info, "update",
            smbs2obsd=parameter_dict["smbs2obsd"], pooling2obsd=parameter_dict["pooling2obsd"],
            obsz2obsd=jsonizable(Obsz2Obsd(**msg_json[0]))
        )

        # Compute the total consumption and generation
        P_gen_pred, P_con_pred = [], []
        for deviceStatusSeries in parameter_dict["obsd2obsz"]["deviceStatusSeriesList"]:
            if deviceStatusSeries["deviceType"] == "load":
                P_con_pred.append(deviceStatusSeries["dataSeries"])
            elif deviceStatusSeries["deviceType"] == "solar" or deviceStatusSeries["deviceType"] == "wind" or deviceStatusSeries["deviceType"] == "chp":
                P_gen_pred.append(deviceStatusSeries["dataSeries"])
        P_gen_pred = list(np.sum(P_gen_pred, axis=0)) if len(P_gen_pred) != 0 else [0 for _ in range(PREDICTION_HORIZON_IN_HOURS * RESOLUTION)]
        P_con_pred = list(np.sum(P_con_pred, axis=0)) if len(P_con_pred) != 0 else [0 for _ in range(PREDICTION_HORIZON_IN_HOURS * RESOLUTION)]

        # Refine the total consumption and generation of next interval by subtracting the pooling results
        if "p2p_trades" in parameter_dict:
            for match in parameter_dict["p2p_trades"]:
                if prosumerId == match["producerID"]:
                    if match["energyKwh"] * RESOLUTION > P_gen_pred[0]:
                        P_con_pred[0] = P_con_pred[0] + (match["energyKwh"] * RESOLUTION - P_gen_pred[0])
                        P_gen_pred[0] = 0
                        logging.warning(
                            f"You may P2P sell more energy than your generation for next interval!"
                            f"Forced to increase power consumption and drain battery buffer!"
                        )
                        logging.warning(f"Generation power: {P_gen_pred[0]}, Sold power: {match['energyKwh'] * RESOLUTION}")
                    else:
                        P_gen_pred[0] = P_gen_pred[0] - match["energyKwh"] * RESOLUTION
                elif prosumerId == match["consumerID"]:
                    if match["energyKwh"] * RESOLUTION > P_con_pred[0]:
                        P_gen_pred[0] = P_gen_pred[0] + (match["energyKwh"] * RESOLUTION - P_con_pred[0])
                        P_con_pred[0] = 0
                        logging.warning(
                            f"You may P2P buy more energy than your consumption for next interval!"
                            f"Forced to increase power generation and charge battery buffer!"
                        )
                        logging.warning(
                            f"Consumed power: {P_con_pred[0]}, bought power: {match['energyKwh'] * RESOLUTION}")
                    else:
                        P_con_pred[0] = P_con_pred[0] - match["energyKwh"] * RESOLUTION

        # Refine the total consumption and generation of the trading interval by subtracting the pooling results
        for match in parameter_dict["pooling2obsd"]["matches"]:
            if prosumerId == match["producerID"]:
                if match["energyKwh"] * RESOLUTION > P_gen_pred[1]:
                    P_con_pred[1] = P_con_pred[1] + (match["energyKwh"] * RESOLUTION - P_gen_pred[1])
                    P_gen_pred[1] = 0
                    logging.warning(
                        f"You may P2P sell more energy than your generation for overnext interval!"
                        f"Forced to increase power consumption and drain battery buffer!"
                    )
                    logging.warning(f"Generation power: {P_gen_pred[1]}, Sold power: {match['energyKwh'] * RESOLUTION}")
                else:
                    P_gen_pred[1] = P_gen_pred[1] - match["energyKwh"] * RESOLUTION
            elif prosumerId == match["consumerID"]:
                if match["energyKwh"] * RESOLUTION > P_con_pred[1]:
                    P_gen_pred[1] = P_gen_pred[1] + (match["energyKwh"] * RESOLUTION - P_con_pred[1])
                    P_con_pred[1] = 0
                    logging.warning(
                        f"You may P2P buy more energy than your consumption for overnext interval!"
                        f"Forced to increase power generation and charge battery buffer!"
                    )
                    logging.warning(
                        f"Consumed power: {P_con_pred[1]}, bought power: {match['energyKwh'] * RESOLUTION}")
                else:
                    P_con_pred[1] = P_con_pred[1] - match["energyKwh"] * RESOLUTION
        parameter_dict["p2p_trades"] = parameter_dict["pooling2obsd"]["matches"]

        loc_idx = parameter_dict["obsz2obsd"]["grid_locations"].index(str(user_config["gridLocation"]))
        # Do the scheduling when the battery exists
        if "battery" in device_types:
            battery_config = user_config["devices"][device_types.index("battery")]

            # Compute the cost
            C_ask = parameter_dict["obsz2obsd"]["predictedClearingPrice"]["dataSeries"]
            C_bid = parameter_dict["obsz2obsd"]["predictedClearingPrice"]["dataSeries"]
            grid_fee = round(max(parameter_dict["obsz2obsd"]["grid_fees"][loc_idx]), 3)  # TODO: static grid fees
            C_grid = [grid_fee for _ in range(PREDICTION_HORIZON_IN_HOURS * RESOLUTION)]

            # Unpack the cleared the results
            P_market2u, P_u2market = 0, 0
            for trade in parameter_dict["smbs2obsd"]["trades"]:
                if trade["energyUnit"] != "kWh":
                    raise NotImplementedError(f"Energy unit {trade['energyUnit']} not implemented!")
                else:
                    if trade["orderType"] == "bid":
                        P_market2u = P_market2u + trade["energy"] * RESOLUTION
                    elif trade["orderType"] == "ask":
                        P_u2market = P_u2market + trade["energy"] * RESOLUTION

            # Get the init battery charging level
            for deviceStatusSeries in parameter_dict["openems2obsd"]["deviceStatusSeriesList"]:
                if deviceStatusSeries["deviceType"] == "battery":
                    if deviceStatusSeries["measuringUnit"] != "kWh":
                        raise NotImplementedError(f"Energy unit {deviceStatusSeries['measuringUnit']} not implemented!")
                    E_bat_init = deviceStatusSeries["dataSeries"][1]

            if if_unit_test():
                Bat_status = 0.5
                ToPowerInkW = 0
                charging_power_in_kW = 0
                FromPowerInkW = 0
                discharging_power_in_kW = 0
                bid_power_in_kW = 0
                ask_power_in_kW = 0
            else:
                if "schedule" in parameter_dict and parameter_dict["schedule"]["charging_power_in_kW"] - parameter_dict["schedule"]["discharging_power_in_kW"] > 0:
                    intially_charing = 1
                else:
                    intially_charing = 0

                from . import scheduler
                order_type, ask_power_in_kW, bid_power_in_kW, \
                    Bat_status, charging_power_in_kW, discharging_power_in_kW, \
                    Xto, ToPowerInkW, Xfrom, FromPowerInkW = \
                    scheduler.schedule(
                        # configuration
                        P_bat_max=battery_config["chargePower"],
                        E_bat=battery_config["capacity"],
                        # current device status
                        E_bat_init=E_bat_init,
                        intially_charing=intially_charing,
                        # status update
                        C_ask=C_ask,
                        C_bid=C_bid,
                        C_grid=C_grid,
                        P_market2u=P_market2u,
                        P_u2market=P_u2market,
                        P_gen_pred=P_gen_pred,
                        P_con_pred=P_con_pred
                    )
                if abs(P_gen_pred[0] + discharging_power_in_kW + FromPowerInkW + P_market2u - (P_con_pred[0] + charging_power_in_kW + ToPowerInkW + P_u2market)) > .1:
                    LOGGER.info(f"P_gen_pred[0]: {P_gen_pred[0]}")
                    LOGGER.info(f"discharging_power_in_kW: {discharging_power_in_kW}")
                    LOGGER.info(f"FromPowerInkW: {FromPowerInkW}")
                    LOGGER.info(f"P_market2u: {P_market2u}")
                    LOGGER.info(f"P_con_pred[0]: {P_con_pred[0]}")
                    LOGGER.info(f"charging_power_in_kW: {charging_power_in_kW}")
                    LOGGER.info(f"ToPowerInkW: {ToPowerInkW}")
                    LOGGER.info(f"P_u2market: {P_u2market}")
                    raise ValueError
            parameter_dict["schedule"] = {
                "bat_status": Bat_status,
                "charging_power_in_kW": ToPowerInkW + charging_power_in_kW,
                "discharging_power_in_kW": FromPowerInkW + discharging_power_in_kW
            }
            if E_bat_init > battery_config["capacity"] * 0.9 - 2 * battery_config["chargePower"] / RESOLUTION:
                print(f"Enforcing discharging! E_bat_int: {E_bat_init}")
                traded_energy = (battery_config["chargePower"] + P_gen_pred[1] - P_con_pred[1]) / RESOLUTION
                if traded_energy > 0:
                    order_type = "ask"
                else:
                    order_type = "bid"
                    traded_energy = -traded_energy
            elif E_bat_init < battery_config["capacity"] * 0.1 + 2 * battery_config["chargePower"] / RESOLUTION:
                print("Enforcing charging! E_bat_int: {E_bat_init}")
                traded_energy = (P_con_pred[1] + battery_config["chargePower"] - P_gen_pred[1]) / RESOLUTION
                if traded_energy > 0:
                    order_type = "bid"
                else:
                    order_type = "ask"
                    traded_energy = -traded_energy
            else:
                if bid_power_in_kW > 0:
                    order_type = "ask"
                    traded_energy = bid_power_in_kW / RESOLUTION
                if ask_power_in_kW >= 0:
                    order_type = "bid"
                    traded_energy = ask_power_in_kW / RESOLUTION
        else:
            logging.info(P_gen_pred)
            logging.info(P_con_pred)
            order_type = "ask" if P_gen_pred[1] > P_con_pred[1] else "bid"
            traded_energy = abs((P_gen_pred[1] - P_con_pred[1]) / RESOLUTION)
        if order_type == "bid":
            price_rate = 0.2 + round(max(parameter_dict["obsz2obsd"]["grid_fees"][loc_idx]) + .001, 3)
        elif order_type =="ask":
            price_rate = 0.0819

        parameter_dict["order"] = [{"order_type": order_type, "traded_energy": traded_energy, "price_rate": price_rate}]

        if "battery" in device_types:
            print(parameter_dict["schedule"])
        print(parameter_dict["order"])


def write_db(simulator_info: SimulatorInfo, handler_type: str, **kwarg):
    if (if_real() or if_server()) and not if_unit_test():
        if simulator_info.action == "individual_power_prediction":
            if simulator_info.interface is None:
                if handler_type == "update":
                    total_gen, total_consum, bat_chr = 0, 0, 0
                    for deviceStatusSeries in kwarg["openems2obsd"]["deviceStatusSeriesList"]:
                        if deviceStatusSeries["deviceType"] in ["wind", "solar"]:
                            total_gen += deviceStatusSeries["dataSeries"][-1] / RESOLUTION
                        elif deviceStatusSeries["deviceType"] == "load":
                            total_consum += deviceStatusSeries["dataSeries"][-1] / RESOLUTION
                        elif deviceStatusSeries["deviceType"] == "battery":
                            bat_chr += deviceStatusSeries["dataSeries"][-1] - deviceStatusSeries["dataSeries"][-2]
                    if total_gen > total_consum + bat_chr:
                        totalEnergyFeedIn = total_gen - (total_consum + bat_chr)
                        totalEnergyExtraction = 0
                    else:
                        totalEnergyFeedIn = 0
                        totalEnergyExtraction = total_consum + bat_chr - total_gen
                    insert_gui_db(
                        "own_energy",
                        {
                            "prosumerId": prosumerId,
                            "timeSlot": infer_last_interval_start(simulator_info)[1],
                            "totalBatteryCharge": bat_chr if bat_chr > 0 else 0,
                            "totalBatteryDischarge": -bat_chr if bat_chr < 0 else 0,
                            "totalEnergyGeneration": round(total_gen, 3),
                            "totalEnergyConsumption": round(total_consum, 3),
                            "totalEnergyFeedIn": round(totalEnergyFeedIn, 3),
                            "totalEnergyExtraction": round(totalEnergyExtraction, 3),
                            "measuringUnit": "kWh"
                        }
                    )
        elif simulator_info.action == "p2p_clearing":
            if simulator_info.interface == "obsd2pooling":
                if handler_type == "get":
                    if kwarg["json"]["greyEnergyKwh"] + kwarg["json"]["greenEnergyKwh"] > 0:
                        order_type = "ask"
                    else:
                        order_type = "bid"
                    if if_server():
                        simulation_time, simulated_time = get_simulation_time(simulator_info)
                        energy = kwarg["json"]["greyEnergyKwh"] + kwarg["json"]["greenEnergyKwh"]
                        if abs(energy) > 10e-7:
                            if energy > 0:
                                for energy_dist in user_config["energyDistributionAsProx"]:
                                    if energy_dist["percentage"] > 10e-7:
                                        insert_metric_db(
                                            "p2p_orders",
                                            {
                                                "simulation_time": simulation_time,
                                                "simulated_time": simulated_time,
                                                "prosumer_id": prosumerId,
                                                "market_time_slot": infer_cur_interval_start(simulator_info)[1],
                                                "step_duration": f"PT{int(60 / RESOLUTION)}M",
                                                "pool": energy_dist["poolId"],
                                                "order_type": order_type,
                                                "green_energy": round(kwarg["json"]["greenEnergyKwh"], 3),
                                                "grey_energy": round(kwarg["json"]["greyEnergyKwh"], 3),
                                                "energy": round(energy * energy_dist["percentage"] / 100, 3),
                                                "market_price_rate": round(kwarg["json"]["marketPriceEurocentPerKwh"] * .01, 3),
                                                "price_unit": "EUR/kWh"
                                            }
                                        )
                            else:
                                for energy_dist in user_config["energyDistributionAsCons"]:
                                    if energy_dist["percentage"] > 10e-7:
                                        insert_metric_db(
                                            "p2p_orders",
                                            {
                                                "simulation_time": simulation_time,
                                                "simulated_time": simulated_time,
                                                "prosumer_id": prosumerId,
                                                "market_time_slot": infer_cur_interval_start(simulator_info)[1],
                                                "step_duration": f"PT{int(60 / RESOLUTION)}M",
                                                "pool": energy_dist["poolId"],
                                                "order_type": order_type,
                                                "green_energy": round(kwarg["json"]["greenEnergyKwh"], 3),
                                                "grey_energy": round(kwarg["json"]["greyEnergyKwh"], 3),
                                                "energy": round(energy * energy_dist["percentage"] / 100, 3),
                                                "market_price_rate": round(kwarg["json"]["marketPriceEurocentPerKwh"] * .01, 3),
                                                "price_unit": "EUR/kWh"
                                            }
                                        )
        elif simulator_info.action == "scheduling":
            if simulator_info.interface == "obsd2smbs" or simulator_info.interface == "agent2smbs":
                if handler_type == "get":
                    if if_server():
                        simulation_time, simulated_time = get_simulation_time(simulator_info)
                        for order in kwarg["json"]["orders"]:
                            insert_metric_db(
                                "orders",
                                {
                                    "simulation_time": simulation_time,
                                    "simulated_time": simulated_time,
                                    "prosumer_id": prosumerId,
                                    "market_time_slot": kwarg["json"]["marketTimeSlot"],
                                    "start_time": infer_historical_interval_start(simulator_info, -2)[1],
                                    "step_duration": f"PT{int(60 / RESOLUTION)}M",
                                    "order_type": order["orderType"],
                                    "energy": round(order["energy"], 3),
                                    "cluster_name": str(user_config["gridLocation"]),
                                    "price_rate": round(order["priceRate"], 3),
                                    "energy_unit": order["energyUnit"],
                                    "price_unit": order["priceUnit"],
                                }
                            )
                    min_smbs_selling_price, smbs_selling_price, max_smbs_buying_price, smbs_buying_price = 0, 0, 0, 0
                    for order_type in ["bid", "ask"]:
                        energies = []
                        prices = []
                        for order in kwarg["json"]["orders"]:
                            if order["orderType"] == order_type:
                                energies.append(order["energy"])
                                prices.append(order["priceRate"])
                        if len(energies) != 0:
                            energies, prices = np.array(energies), np.array(prices)
                            if order_type == "bid":
                                min_smbs_selling_price = prices.min()
                                smbs_selling_price = \
                                    (prices * energies).sum() / energies.sum() if abs(energies.sum()) > 10e-7 else 0
                            else:
                                max_smbs_buying_price = prices.max()
                                smbs_buying_price = \
                                    (prices * energies).sum() / energies.sum() if abs(energies.sum()) > 10e-7 else 0
                    insert_gui_db(
                        "own_price",
                        {
                            "prosumerId": prosumerId,
                            "timeSlot": infer_cur_interval_start(simulator_info)[1],
                            "maxSMBSBuyingPrice": round(max_smbs_buying_price, 3),
                            "SMBSBuyingPrice": round(smbs_buying_price, 3),
                            "minSMBSSellingPrice": round(min_smbs_selling_price, 3),
                            "SMBSSellingPrice": round(smbs_selling_price, 3),
                            "energyUnit": kwarg["json"]["orders"][0]["energyUnit"],
                            "priceUnit": kwarg["json"]["orders"][0]["priceUnit"]
                        }
                    )
            elif simulator_info.interface is None:
                if handler_type == "update":
                    for trade in kwarg["smbs2obsd"]["trades"]:
                        if if_server():
                            simulation_time, simulated_time = get_simulation_time(simulator_info)
                            insert_metric_db(
                                "trades",
                                {
                                    "simulation_time": simulation_time,
                                    "simulated_time": simulated_time,
                                    "prosumer_id": prosumerId,
                                    "market_time_slot": kwarg["smbs2obsd"]["marketTimeSlot"],
                                    "start_time": infer_historical_interval_start(simulator_info, -1)[1],
                                    "step_duration": f"PT{int(60 / RESOLUTION)}M",
                                    "order_type": trade["orderType"],
                                    "energy": round(trade["energy"], 3),
                                    "price_rate": round(trade["priceRate"], 3),
                                    "cluster_name": trade["clusterName"],
                                    "energy_unit": trade["energyUnit"],
                                    "price_unit": trade["priceUnit"]
                                }
                            )
                        try:
                            insert_gui_db(
                            "trades",
                            {
                                "prosumerId": prosumerId,
                                "marketTimeSlot": kwarg["smbs2obsd"]["marketTimeSlot"],
                                "stepDuration": f"PT{int(60 / RESOLUTION)}M",
                                "orderType": trade["orderType"],
                                "energy": trade["energy"],
                                "priceRate": trade["priceRate"],
                                "clusterName": trade["clusterName"],
                                "energyUnit": trade["energyUnit"],
                                "priceUnit": trade["priceUnit"]
                            }
                        )
                        except:
                            print("Entry may exists!")
                    for trade in kwarg["pooling2obsd"]["matches"]:
                        if if_server():
                            simulation_time, simulated_time = get_simulation_time(simulator_info)
                            entry = {
                                "simulation_time": simulation_time,
                                "simulated_time": simulated_time,
                                "prosumer_id": prosumerId,
                                "market_time_slot": infer_cur_interval_start(simulator_info)[1],
                                "step_duration": f"PT{int(60 / RESOLUTION)}M",
                                "order_type": "bid" if trade["producerID"] != prosumerId else "ask",
                                "producer_id": trade["producerID"],
                                "consumer_id": trade["consumerID"],
                                # "trading_partner_id": trade["producerID"] if trade["producerID"] != prosumerId else trade["consumerID"],
                                "pool_id": trade["poolID"],
                                "energy": round(trade["energyKwh"], 3),
                                "pool_price_rate": round(trade["energyPoolPrice"], 3),
                                "grid_usage_fee": round(trade["gridUsageFee"], 3),
                                "price_unit": "EUR/kWh",
                                "leftover_energy": round(kwarg["pooling2obsd"]["energyKwh"], 3)
                            }
                            try:
                                if abs(entry["energy"]) > 10e-7:
                                    insert_metric_db("p2p_trades", entry)
                            except:
                                print("Entry may exists!")
                                print(entry)
                        try:
                            if abs(entry["energy"]) > 10e-7:
                                insert_gui_db(
                                    "own_energy_detail_pooling",
                                    {
                                        "prosumerId": prosumerId,
                                        "timeSlot": infer_cur_interval_start(simulator_info)[1],
                                        "producerId": trade["producerID"],
                                        "consumerId": trade["consumerID"],
                                        "poolId": trade["poolID"],
                                        "orderType": "bid" if trade["producerID"] != prosumerId else "ask",
                                        "energy": round(trade["energyKwh"], 3),
                                        "leftoverEnergy": round(kwarg["pooling2obsd"]["energyKwh"], 3),
                                        "poolingPriceRate": round(trade["energyPoolPrice"], 3),
                                        "measuringUnit": "EUR/kWh"
                                    }
                                )
                        except:
                            print("Entry may exists!")
                            print(entry)

                    prices, energies = [], []
                    for trade in kwarg["pooling2obsd"]["matches"]:
                        if prosumerId == trade["producerID"]:
                            energies.append(trade["energyKwh"])
                            prices.append(trade["energyPoolPrice"])
                    if len(prices) != 0:
                        energies, prices = np.array(energies), np.array(prices)
                        insert_gui_db(
                            "own_price_detail_pooling",
                            {
                                      "prosumerId": prosumerId,
                                      "timeSlot": infer_cur_interval_start(simulator_info)[1],
                                      "orderType": "bid",
                                      "energy": round(energies.sum(), 3),
                                      "energyWeightedPrice": round((prices * energies).sum() / energies.sum(), 3),
                                      "measuringUnit": "EUR/kWh"
                            }
                        )

                    prices, energies = [], []
                    for trade in kwarg["pooling2obsd"]["matches"]:
                        if prosumerId == trade["consumerID"]:
                            energies.append(trade["energyKwh"])
                            prices.append(trade["energyPoolPrice"])
                    if len(prices) != 0:
                        energies, prices = np.array(energies), np.array(prices)
                        insert_gui_db(
                            "own_price_detail_pooling",
                            {
                                "prosumerId": prosumerId,
                                "timeSlot": infer_cur_interval_start(simulator_info)[1],
                                "orderType": "ask",
                                "energy": energies.sum(),
                                "energyWeightedPrice": (prices * energies).sum() / energies.sum(),
                                "measuringUnit": "EUR/kWh"
                            }
                        )

                    for summary in kwarg["obsz2obsd"]["summaries"]:
                        insert_gui_db(
                            "best_energy_price",
                            {
                                "marketTimeSlot": summary["marketTimeSlot"],
                                "clusterOrigin": summary["clusterOrigin"],
                                "clusterDestination": summary["clusterDestination"],
                                "bidEnergySum": round(summary.get("bidEnergySum", 0), 3),
                                "askEnergySum": round(summary.get("askEnergySum", 0), 3),
                                "clearedEnergySum": round(summary["clearedEnergySum"], 3),
                                "priceRateWeightedAverage": round(summary.get("priceRateWeightedAverage", 0), 3),
                                "energyUnit": summary["energyUnit"],
                                "priceUnit": summary["priceUnit"]
                            }
                        )


@obsd_client.put("/api/obsd/individual_power_prediction")
async def individual_power_prediction(simulator_info: SimulatorInfo):
    pre_handlers = [requests.put, requests.get, requests.put]
    pre_handler_args = [
        {"url": f"http://{LOCALHOST}:{OPENEMS_PORT}/api/openems/openems2obsd"},
        {"url": f"http://{LOCALHOST}:{COORDINATOR_PORT}/api/bws/bws2obsd"},
        {"url": f"http://{LOCALHOST}:{COORDINATOR_PORT}/api/obsz/obsz2obsd"}
    ]
    post_handlers = [requests.post]
    post_handler_args = [{"url": f"http://{LOCALHOST}:{COORDINATOR_PORT}/api/obsz/obsd2obsz"}]

    return action_handler(
        simulator_info, OBSD_LOG, update, get,
        pre_interface_handlers=pre_handlers, pre_interface_handler_args=pre_handler_args,
        post_interface_handlers=post_handlers, post_interface_handler_args=post_handler_args
    )


@obsd_client.put('/api/obsd/scheduling')
async def scheduling(simulator_info: SimulatorInfo):
    pre_interface_handlers = [requests.put, requests.put, requests.put]
    pre_interface_handler_args = [
        {"url": f"http://{LOCALHOST}:{COORDINATOR_PORT}/api/obsz/obsz2obsd"},
        {"url": f"http://{LOCALHOST}:{COORDINATOR_PORT}/api/smbs/smbs2obsd"},
        {"url": f"http://{LOCALHOST}:{POOLING_PORT}/api/pooling/pooling2obsd"},
    ]
    post_interface_handlers = [requests.post, requests.post]
    post_interface_handler_args = [
        {"url": f"http://{LOCALHOST}:{COORDINATOR_PORT}/api/smbs/agent2smbs"},
        {"url": f"http://{LOCALHOST}:{OPENEMS_PORT}/api/openems/obsd2openems"}
    ]

    return action_handler(
        simulator_info, OBSD_LOG, update, get,
        pre_interface_handlers=pre_interface_handlers, pre_interface_handler_args=pre_interface_handler_args,
        post_interface_handlers=post_interface_handlers, post_interface_handler_args=post_interface_handler_args
    )


@obsd_client.put('/api/obsd/send_measurement')
async def send_measurement(simulator_info: SimulatorInfo):
    post_interface_handlers = [requests.post]
    post_interface_handler_args = [{"url": f"http://{LOCALHOST}:{COORDINATOR_PORT}/api/bkv/obsd2bkv"}]

    return action_handler(
        simulator_info, OBSD_LOG, update, get,
        post_interface_handlers=post_interface_handlers, post_interface_handler_args=post_interface_handler_args
    )


@obsd_client.put('/api/obsd/obsd2pooling')
async def obsd2pooling(simulator_info: SimulatorInfo):
    return interface_handler(
        simulator_info, received_msg_json=None, log_file=OBSD_LOG, update_handler=update, get_handler=get
    )


def obsd():
    LogConfig.startup_info("OBSD")
    uvicorn.run(obsd_client, host=LOCALHOST, port=OBSD_PORT, log_config=None)
