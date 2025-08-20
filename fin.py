# -*- coding: utf-8 -*-
# auto-obfuscated (abbrev-aware, v3):
# - Renames function keyword-only/named params and updates call-site keywords for local functions/methods
# - Renames lambda parameters (args/kwonly/vararg/kwarg)
# - Keeps self/cls attribute renaming; skips external APIs

import asyncio
from copy import copy
import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
import threading
import time
import math
import json
import base64
import os
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    UploadFile,
    File,
)
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import easyocr
import requests
import concurrent.futures
import sys

sys.path.append("./soyo")
import onnxruntime as ort
from ina226 import INA226

v_vis: Optional[np.ndarray] = None
v_msk: Optional[np.ndarray] = None
v_watimg: Optional[np.ndarray] = None
v_msi: List[np.ndarray] = []
v_shueve: threading.Event = threading.Event()
v_capthr: Optional[threading.Thread] = None
v_app: FastAPI = FastAPI()
v_ina: INA226
print(eval('str("DEBUG")'))
try:
    v_ina = INA226(i2c_bus=1, address=64, shunt_ohms=0.01, max_expected_amps=5.0)
    print(eval('str("DEBUG")'))
    print(eval('str("DEBUG")'))
    v_mea_2 = v_ina.get_all_measurements()
    print(eval('str("DEBUG")'))
    print(eval('str("DEBUG")'))
    print(eval('str("DEBUG")'))
    print(eval('str("DEBUG")'))
    print(eval('str("DEBUG")'))
except Exception as v_e:
    print(eval('str("DEBUG")'))
    v_ina = None
v_app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
CONFIG_FILE: str = "config.json"
v_capo = True
OCR_LANGS = ["la"]
OCR_ALLOWLIST = "0123456789"
USE_GPU = False
v_rea = None
v_asy = None
v_locsoyo = None
v_rawcro: List[Any]


class cls_locsoyo:
    def __init__(self, v_mdlp="./soyo/best.onnx"):
        self.v_net = None
        self.v_mdlh = 320
        self.v_mdlw = 320
        self.v_nl = 3
        self.v_na = 3
        self.v_strd = [8.0, 16.0, 32.0]
        self.v_ancgri = None
        self.v_diclab = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8"}
        self.v_mdlp = v_mdlp
        self.fn_loamdl()

    def fn_makgri(self, v_nx, v_ny):
        v_xv, v_yv = np.meshgrid(np.arange(v_ny), np.arange(v_nx))
        return np.stack((v_xv, v_yv), 2).reshape((-1, 2)).astype(np.float32)

    def fn_calout(self, v_out_2, v_nl, v_na, v_mdlw, v_mdlh, v_ancgri, v_strd):
        v_rowind = 0
        v_gri = [np.zeros(1)] * v_nl
        for v_i in range(v_nl):
            v_h, v_w = (int(v_mdlw / v_strd[v_i]), int(v_mdlh / v_strd[v_i]))
            v_len = int(v_na * v_h * v_w)
            if v_gri[v_i].shape[2:4] != (v_h, v_w):
                v_gri[v_i] = self.fn_makgri(v_w, v_h)
            v_out_2[v_rowind : v_rowind + v_len, 0:2] = (
                v_out_2[v_rowind : v_rowind + v_len, 0:2] * 2.0
                - 0.5
                + np.tile(v_gri[v_i], (v_na, 1))
            ) * int(v_strd[v_i])
            v_out_2[v_rowind : v_rowind + v_len, 2:4] = (
                v_out_2[v_rowind : v_rowind + v_len, 2:4] * 2
            ) ** 2 * np.repeat(v_ancgri[v_i], v_h * v_w, axis=0)
            v_rowind += v_len
        return v_out_2

    def fn_ppo(self, v_out, v_mdlh, v_mdlw, v_imgh, v_imgw, v_thrnms, v_thrcon):
        v_con = v_out[:, 4].tolist()
        v_cx = v_out[:, 0] / v_mdlw * v_imgw
        v_cy = v_out[:, 1] / v_mdlh * v_imgh
        v_w = v_out[:, 2] / v_mdlw * v_imgw
        v_h = v_out[:, 3] / v_mdlh * v_imgh
        v_pcls = v_out[:, 5:]
        if len(v_pcls.shape) == 1:
            v_pcls = np.expand_dims(v_pcls, 1)
        v_clsid = np.argmax(v_pcls, axis=1)
        v_px1 = np.expand_dims(v_cx - v_w / 2, -1)
        v_py1 = np.expand_dims(v_cy - v_h / 2, -1)
        v_px2 = np.expand_dims(v_cx + v_w / 2, -1)
        v_py2 = np.expand_dims(v_cy + v_h / 2, -1)
        v_are_2 = np.concatenate((v_px1, v_py1, v_px2, v_py2), axis=-1)
        v_are_2 = v_are_2.tolist()
        v_ids = cv2.dnn.NMSBoxes(v_are_2, v_con, v_thrcon, v_thrnms)
        if len(v_ids) > 0:
            return (np.array(v_are_2)[v_ids], np.array(v_con)[v_ids], v_clsid[v_ids])
        else:
            return ([], [], [])

    def fn_loamdl(self):
        try:
            v_so = ort.SessionOptions()
            self.v_net = ort.InferenceSession(self.v_mdlp, v_so)
            v_anc = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
            self.v_ancgri = np.asarray(v_anc, dtype=np.float32).reshape(
                self.v_nl, -1, 2
            )
            print(eval('str("DEBUG")'))
            return True
        except Exception as v_e:
            print(eval('str("DEBUG")'))
            return False

    def fn_pre(self, v_img_2, v_thrnms=0.4, v_thrcon=0.5):
        if self.v_net is None:
            print(eval('str("DEBUG")'))
            return {"success": False, "count": 0, "results": []}
        try:
            v_imgpro = cv2.resize(
                v_img_2, [self.v_mdlw, self.v_mdlh], interpolation=cv2.INTER_AREA
            )
            v_imgpro = cv2.cvtColor(v_imgpro, cv2.COLOR_BGR2RGB)
            v_imgpro = v_imgpro.astype(np.float32) / 255.0
            v_blo = np.expand_dims(np.transpose(v_imgpro, (2, 0, 1)), axis=0)
            v_out_2 = self.v_net.run(None, {self.v_net.get_inputs()[0].name: v_blo})[
                0
            ].squeeze(axis=0)
            v_out_2 = self.fn_calout(
                v_out_2,
                self.v_nl,
                self.v_na,
                self.v_mdlw,
                self.v_mdlh,
                self.v_ancgri,
                self.v_strd,
            )
            v_imgh, v_imgw, v__ = np.shape(v_img_2)
            v_bxs, v_con_3, v_ids = self.fn_ppo(
                v_out_2, self.v_mdlh, self.v_mdlw, v_imgh, v_imgw, v_thrnms, v_thrcon
            )
            v_res = []
            for v_box, v_con, id in zip(v_bxs, v_con_3, v_ids):
                v_res_2 = {
                    "label": self.v_diclab[id],
                    "confidence": float(v_con),
                    "bbox": {
                        "x1": float(v_box[0]),
                        "y1": float(v_box[1]),
                        "x2": float(v_box[2]),
                        "y2": float(v_box[3]),
                    },
                }
                v_res.append(v_res_2)
            return {
                "success": True,
                "count": len(v_res),
                "results": v_res,
                "image_shape": {
                    "height": v_img_2.shape[0],
                    "width": v_img_2.shape[1],
                    "channels": v_img_2.shape[2],
                },
            }
        except Exception as v_e:
            print(eval('str("DEBUG")'))
            return {"success": False, "count": 0, "results": [], "error": str(v_e)}


def fn_getrea():
    global v_rea
    if v_rea is None:
        v_rea = easyocr.Reader(OCR_LANGS, gpu=USE_GPU)
    return v_rea


def fn_gly():
    global v_locsoyo
    if v_locsoyo is None:
        v_locsoyo = cls_locsoyo()
    return v_locsoyo


def fn_gasy():
    global v_asy
    if v_asy is None:
        try:
            v_asy = cls_lasy()
            print(eval('str("DEBUG")'))
        except Exception as v_e:
            print(eval('str("DEBUG")'))
            v_asy = None
    return v_asy


class cls_laseo:
    def __init__(self):
        self.v_res = []

    def fn_ewr(self, v_img_2, v_mnare=300):
        if len(v_img_2.shape) == 3 and v_img_2.shape[2] == 3:
            v_gry = cv2.cvtColor(v_img_2, cv2.COLOR_BGR2GRAY)
        elif len(v_img_2.shape) == 3 and v_img_2.shape[2] == 1:
            v_gry = v_img_2[:, :, 0]
        else:
            v_gry = v_img_2
        v__, v_bin = cv2.threshold(v_gry, 200, 255, cv2.THRESH_BINARY)
        v_ker = np.ones((3, 3), np.uint8)
        v_bin = cv2.morphologyEx(v_bin, cv2.MORPH_CLOSE, v_ker, iterations=2)
        v_bin = cv2.morphologyEx(v_bin, cv2.MORPH_OPEN, v_ker, iterations=1)
        v_con_5, v__ = cv2.findContours(
            v_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        v_rec_2 = []
        for v_i, v_con_4 in enumerate(v_con_5):
            v_are = cv2.contourArea(v_con_4)
            if v_are < v_mnare:
                continue
            v_eps = 0.02 * cv2.arcLength(v_con_4, True)
            v_app_2 = cv2.approxPolyDP(v_con_4, v_eps, True)
            if len(v_app_2) != 4:
                v_rec = cv2.minAreaRect(v_con_4)
                v_box = cv2.boxPoints(v_rec)
                v_box = np.array(v_box, dtype=np.int32)
                v_app_2 = v_box.reshape(-1, 1, 2)
            v_recare = cv2.contourArea(v_app_2)
            if v_recare == 0:
                continue
            v_arerat = v_are / v_recare
            if v_arerat < 0.5:
                continue
            v_corimg, v_bio = self.fn_eacr(v_img_2, v_app_2.reshape(4, 2))
            if v_corimg is not None:
                v_h, v_w = v_corimg.shape[:2]
                if v_w >= 20 and v_h >= 20:
                    v_asprat = v_w / v_h
                    if 0.1 <= v_asprat <= 10:
                        v_rec_2.append(
                            {
                                "id": v_i,
                                "image": v_corimg,
                                "bbox_in_original": v_bio,
                                "contour_points": v_app_2.reshape(4, 2),
                                "area": v_are,
                                "corrected_size": (v_w, v_h),
                            }
                        )
        return v_rec_2

    def fn_eacr(self, v_img_2, v_pts):
        try:
            v_pts = np.array(v_pts, dtype=np.float32)
            if v_pts.shape != (4, 2):
                return (None, None)
            v_ordpts = self.fn_ordpts(v_pts)
            v_w_2, v_h_2 = self.fn_ccs(v_ordpts)
            if v_w_2 < 20 or v_h_2 < 20:
                return (None, None)
            v_dstpts = np.array(
                [[0, 0], [v_w_2 - 1, 0], [v_w_2 - 1, v_h_2 - 1], [0, v_h_2 - 1]],
                dtype=np.float32,
            )
            v_tramat = cv2.getPerspectiveTransform(v_ordpts, v_dstpts)
            v_corimg = cv2.warpPerspective(v_img_2, v_tramat, (int(v_w_2), int(v_h_2)))
            v_xcoo = v_ordpts[:, 0]
            v_ycoo = v_ordpts[:, 1]
            v_bio = {
                "x_min": float(np.min(v_xcoo)),
                "y_min": float(np.min(v_ycoo)),
                "x_max": float(np.max(v_xcoo)),
                "y_max": float(np.max(v_ycoo)),
                "transform_matrix": v_tramat,
                "original_points": v_ordpts,
                "corrected_size": (int(v_w_2), int(v_h_2)),
            }
            return (v_corimg, v_bio)
        except Exception as v_e:
            print(eval('str("DEBUG")'))
            return (None, None)

    def fn_ordpts(self, v_pts_2):
        v_ctr = np.mean(v_pts_2, axis=0)

        def fn_afc(v_pt):
            return np.arctan2(v_pt[1] - v_ctr[1], v_pt[0] - v_ctr[0])

        v_sorpts = sorted(v_pts_2, key=fn_afc)
        v_dfo = [v_pt_2[0] + v_pt_2[1] for v_pt_2 in v_sorpts]
        v_staidx = np.argmin(v_dfo)
        v_ord = []
        for v_i in range(4):
            v_ord.append(v_sorpts[(v_staidx + v_i) % 4])
        return np.array(v_ord, dtype=np.float32)

    def fn_ccs(self, v_ordpts):
        v_topw = np.linalg.norm(v_ordpts[1] - v_ordpts[0])
        v_botw = np.linalg.norm(v_ordpts[2] - v_ordpts[3])
        v_lefh = np.linalg.norm(v_ordpts[3] - v_ordpts[0])
        v_righ = np.linalg.norm(v_ordpts[2] - v_ordpts[1])
        v_w_2 = max(v_topw, v_botw)
        v_h_2 = max(v_lefh, v_righ)
        return (int(v_w_2), int(v_h_2))

    def fn_proimg(
        self, v_imgp, v_confthr=0.5, v_nmsthr=0.4, v_mnare=300, v_savdeb=False
    ):
        if isinstance(v_imgp, str):
            v_img_2 = cv2.imread(v_imgp)
        else:
            v_img_2 = v_imgp
        if v_img_2 is None:
            print(eval('str("DEBUG")'))
            return []
        print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        v_outdir = "/tmp/out"
        os.makedirs(v_outdir, exist_ok=True)
        print(eval('str("DEBUG")'))
        v_rec_2 = self.fn_ewr(v_img_2, v_mnare)
        print(eval('str("DEBUG")'))
        v_allres = []
        for v_i, v_recinf in enumerate(v_rec_2):
            print(eval('str("DEBUG")'))
            v_recimg = v_recinf["image"]
            v_bio = v_recinf["bbox_in_original"]
            v_recfil = f"/tmp/out/rect_easyocr_{v_i + 1:02d}.jpg"
            cv2.imwrite(v_recfil, v_recimg)
            print(eval('str("DEBUG")'))
            try:
                v_ocrres_2 = fn_roof(v_recimg)
                if v_ocrres_2:
                    print(eval('str("DEBUG")'))
                    for v_j, v_det in enumerate(v_ocrres_2):
                        print(eval('str("DEBUG")'))
                        v_bboxpts = v_det["bbox"]
                        v_cxc = sum([v_p[0] for v_p in v_bboxpts]) / 4
                        v_cyc = sum([v_p[1] for v_p in v_bboxpts]) / 4
                        v_tramat = v_bio["transform_matrix"]
                        v_invmat = cv2.invert(v_tramat)[1]
                        v_corpt = np.array([v_cxc, v_cyc, 1])
                        v_oript = v_invmat @ v_corpt
                        v_ctrx = v_oript[0] / v_oript[2]
                        v_ctry = v_oript[1] / v_oript[2]
                        v_oribbox = []
                        for v_pt in v_bboxpts:
                            v_corpt_2 = np.array([v_pt[0], v_pt[1], 1])
                            v_oript_2 = v_invmat @ v_corpt_2
                            v_oribbox.append(
                                [
                                    v_oript_2[0] / v_oript_2[2],
                                    v_oript_2[1] / v_oript_2[2],
                                ]
                            )
                        v_res_2 = {
                            "text": v_det["text"],
                            "confidence": v_det["conf"],
                            "center_x": v_ctrx,
                            "center_y": v_ctry,
                            "rectangle_id": v_recinf["id"],
                            "rectangle_area": v_recinf["area"],
                            "corrected_size": v_recinf["corrected_size"],
                            "bbox_in_corrected": v_bboxpts,
                            "bbox_in_original": v_oribbox,
                        }
                        v_allres.append(v_res_2)
                else:
                    print(eval('str("DEBUG")'))
            except Exception as v_e:
                print(eval('str("DEBUG")'))
        v_orifil = f"/tmp/out/original_easyocr.jpg"
        cv2.imwrite(v_orifil, v_img_2)
        print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        if v_allres:
            print(eval('str("DEBUG")'))
            for v_i, v_res_2 in enumerate(v_allres):
                print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        return v_allres


class cls_lasy:
    def __init__(self):
        self.v_soyo = fn_gly()
        self.v_res = []

    def fn_ewr(self, v_img_2, v_mnare=300):
        v_gry = cv2.cvtColor(v_img_2, cv2.COLOR_BGR2GRAY)
        v__, v_bin = cv2.threshold(v_gry, 200, 255, cv2.THRESH_BINARY)
        v_ker = np.ones((3, 3), np.uint8)
        v_bin = cv2.morphologyEx(v_bin, cv2.MORPH_CLOSE, v_ker, iterations=2)
        v_bin = cv2.morphologyEx(v_bin, cv2.MORPH_OPEN, v_ker, iterations=1)
        v_con_5, v__ = cv2.findContours(
            v_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        v_rec_2 = []
        for v_i, v_con_4 in enumerate(v_con_5):
            v_are = cv2.contourArea(v_con_4)
            if v_are < v_mnare:
                continue
            v_eps = 0.02 * cv2.arcLength(v_con_4, True)
            v_app_2 = cv2.approxPolyDP(v_con_4, v_eps, True)
            if len(v_app_2) != 4:
                v_rec = cv2.minAreaRect(v_con_4)
                v_box = cv2.boxPoints(v_rec)
                v_box = np.array(v_box, dtype=np.int32)
                v_app_2 = v_box.reshape(-1, 1, 2)
            v_recare = cv2.contourArea(v_app_2)
            if v_recare == 0:
                continue
            v_arerat = v_are / v_recare
            if v_arerat < 0.5:
                continue
            v_corimg, v_bio = self.fn_eacr(v_img_2, v_app_2.reshape(4, 2))
            if v_corimg is not None:
                v_h, v_w = v_corimg.shape[:2]
                if v_w >= 20 and v_h >= 20:
                    v_asprat = v_w / v_h
                    if 0.1 <= v_asprat <= 10:
                        v_rec_2.append(
                            {
                                "id": v_i,
                                "image": v_corimg,
                                "bbox_in_original": v_bio,
                                "contour_points": v_app_2.reshape(4, 2),
                                "area": v_are,
                                "corrected_size": (v_w, v_h),
                            }
                        )
        return v_rec_2

    def fn_eacr(self, v_img_2, v_pts):
        try:
            v_pts = np.array(v_pts, dtype=np.float32)
            if v_pts.shape != (4, 2):
                return (None, None)
            v_ordpts = self.fn_ordpts(v_pts)
            v_w_2, v_h_2 = self.fn_ccs(v_ordpts)
            if v_w_2 < 20 or v_h_2 < 20:
                return (None, None)
            v_dstpts = np.array(
                [[0, 0], [v_w_2 - 1, 0], [v_w_2 - 1, v_h_2 - 1], [0, v_h_2 - 1]],
                dtype=np.float32,
            )
            v_tramat = cv2.getPerspectiveTransform(v_ordpts, v_dstpts)
            v_corimg = cv2.warpPerspective(v_img_2, v_tramat, (int(v_w_2), int(v_h_2)))
            v_xcoo = v_ordpts[:, 0]
            v_ycoo = v_ordpts[:, 1]
            v_bio = {
                "x_min": float(np.min(v_xcoo)),
                "y_min": float(np.min(v_ycoo)),
                "x_max": float(np.max(v_xcoo)),
                "y_max": float(np.max(v_ycoo)),
                "transform_matrix": v_tramat,
                "original_points": v_ordpts,
                "corrected_size": (int(v_w_2), int(v_h_2)),
            }
            return (v_corimg, v_bio)
        except Exception as v_e:
            print(eval('str("DEBUG")'))
            return (None, None)

    def fn_ordpts(self, v_pts_2):
        v_ctr = np.mean(v_pts_2, axis=0)

        def fn_afc(v_pt):
            return np.arctan2(v_pt[1] - v_ctr[1], v_pt[0] - v_ctr[0])

        v_sorpts = sorted(v_pts_2, key=fn_afc)
        v_dfo = [v_pt_2[0] + v_pt_2[1] for v_pt_2 in v_sorpts]
        v_staidx = np.argmin(v_dfo)
        v_ord = []
        for v_i in range(4):
            v_ord.append(v_sorpts[(v_staidx + v_i) % 4])
        return np.array(v_ord, dtype=np.float32)

    def fn_ccs(self, v_ordpts):
        v_topw = np.linalg.norm(v_ordpts[1] - v_ordpts[0])
        v_botw = np.linalg.norm(v_ordpts[2] - v_ordpts[3])
        v_lefh = np.linalg.norm(v_ordpts[3] - v_ordpts[0])
        v_righ = np.linalg.norm(v_ordpts[2] - v_ordpts[1])
        v_w_2 = max(v_topw, v_botw)
        v_h_2 = max(v_lefh, v_righ)
        return (int(v_w_2), int(v_h_2))

    def fn_proimg(
        self, v_imgp, v_confthr=0.5, v_nmsthr=0.4, v_mnare=300, v_savdeb=False
    ):
        if isinstance(v_imgp, str):
            v_img_2 = cv2.imread(v_imgp)
        else:
            v_img_2 = v_imgp
        if v_img_2 is None:
            print(eval('str("DEBUG")'))
            return []
        print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        v_outdir = "/tmp/out"
        os.makedirs(v_outdir, exist_ok=True)
        print(eval('str("DEBUG")'))
        v_rec_2 = self.fn_ewr(v_img_2, v_mnare)
        print(eval('str("DEBUG")'))
        v_allres = []
        for v_i, v_recinf in enumerate(v_rec_2):
            print(eval('str("DEBUG")'))
            v_recimg = v_recinf["image"]
            v_bio = v_recinf["bbox_in_original"]
            v_recfil = f"/tmp/out/rect_{v_i + 1:02d}.jpg"
            cv2.imwrite(v_recfil, v_recimg)
            print(eval('str("DEBUG")'))
            v_soyores = self.v_soyo.fn_pre(
                v_recimg, v_nmsthr, v_confthr
            )
            if v_soyores.get("success") and v_soyores.get("count", 0) > 0:
                print(eval('str("DEBUG")'))
                for v_j, v_det in enumerate(v_soyores["results"]):
                    print(eval('str("DEBUG")'))
                    v_bbox = v_det["bbox"]
                    v_cxc = (v_bbox["x1"] + v_bbox["x2"]) / 2
                    v_cyc = (v_bbox["y1"] + v_bbox["y2"]) / 2
                    v_tramat = v_bio["transform_matrix"]
                    v_invmat = cv2.invert(v_tramat)[1]
                    v_corpt = np.array([v_cxc, v_cyc, 1])
                    v_oript = v_invmat @ v_corpt
                    v_ctrx = v_oript[0] / v_oript[2]
                    v_ctry = v_oript[1] / v_oript[2]
                    v_res_2 = {
                        "text": v_det["label"],
                        "confidence": v_det["confidence"],
                        "center_x": v_ctrx,
                        "center_y": v_ctry,
                        "rectangle_id": v_recinf["id"],
                        "rectangle_area": v_recinf["area"],
                        "corrected_size": v_recinf["corrected_size"],
                        "bbox_in_corrected": {
                            "x1": v_bbox["x1"],
                            "y1": v_bbox["y1"],
                            "x2": v_bbox["x2"],
                            "y2": v_bbox["y2"],
                        },
                    }
                    v_allres.append(v_res_2)
            else:
                print(eval('str("DEBUG")'))
        v_orifil = f"/tmp/out/original.jpg"
        cv2.imwrite(v_orifil, v_img_2)
        print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        if v_allres:
            print(eval('str("DEBUG")'))
            for v_i, v_res_2 in enumerate(v_allres):
                print(eval('str("DEBUG")'))
        print(eval('str("DEBUG")'))
        return v_allres


def fn_piwas(v_ib6_2, v_idx):
    try:
        print(eval('str("DEBUG")'))
        v_autseg = fn_gasy()
        if v_autseg is None:
            print(eval('str("DEBUG")'))
            return []
        v_imgdat = base64.b64decode(v_ib6_2)
        v_temp = f"/tmp/temp_crop_{v_idx}_{int(time.time())}.jpg"
        with open(v_temp, "wb") as v_f:
            v_f.write(v_imgdat)
        print(eval('str("DEBUG")'))
        v_res = v_autseg.fn_proimg(
            v_temp,
            0.5,
            0.4,
            300,
            True,
        )
        print(eval('str("DEBUG")'))
        if os.path.exists(v_temp):
            os.remove(v_temp)
        v_conres_2 = []
        for v_ite in v_res:
            v_ctrx = v_ite["center_x"]
            v_ctry = v_ite["center_y"]
            v_corsz = v_ite.get("corrected_size", [50, 50])
            v_halw = v_corsz[0] / 4
            v_halh = v_corsz[1] / 4
            v_bbox = [
                [v_ctrx - v_halw, v_ctry - v_halh],
                [v_ctrx + v_halw, v_ctry - v_halh],
                [v_ctrx + v_halw, v_ctry + v_halh],
                [v_ctrx - v_halw, v_ctry + v_halh],
            ]
            v_conres_2.append(
                {
                    "text": v_ite["text"],
                    "conf": v_ite["confidence"],
                    "bbox": v_bbox,
                    "center": [v_ctrx, v_ctry],
                    "rectangle_id": v_ite.get("rectangle_id", 0),
                    "rectangle_area": v_ite.get("rectangle_area", 0),
                }
            )
        print(eval('str("DEBUG")'))
        return v_conres_2
    except Exception as v_e:
        print(eval('str("DEBUG")'))
        import traceback

        traceback.print_exc()
        return []


def fn_roof(v_frm: np.ndarray):
    if v_frm is None or v_frm.size == 0:
        return []
    v_rea_2 = fn_getrea()
    v_rgb = cv2.cvtColor(v_frm, cv2.COLOR_BGR2RGB)
    v_res = v_rea_2.readtext(v_rgb, allowlist=OCR_ALLOWLIST, detail=1)
    v_par_2 = []
    for v_bbox, v_tex, v_con in v_res:
        if OCR_ALLOWLIST and any((v_c not in OCR_ALLOWLIST for v_c in v_tex)):
            continue
        v_par_2.append(
            {
                "text": v_tex,
                "conf": float(v_con),
                "bbox": [list(map(int, v_pt_2)) for v_pt_2 in v_bbox],
            }
        )
    return v_par_2


def fn_loacon() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as v_f:
                return json.load(v_f)
        except Exception as v_e:
            print(eval('str("DEBUG")'))
            return fn_gdc()
    else:
        v_con_2 = fn_gdc()
        fn_savcon(v_con_2)
        return v_con_2


def fn_savcon(v_con_2: Optional[Dict[str, Any]] = None) -> None:
    if v_con_2 is None:
        v_con_2 = {
            "camera": v_camcon,
            "detection": v_par,
            "area_filter": v_afp,
            "perspective_correction": v_perpar,
            "black_detection": v_bdp,
        }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as v_f:
            json.dump(v_con_2, v_f, indent=4, ensure_ascii=False)
        print(eval('str("DEBUG")'))
    except Exception as v_e:
        print(eval('str("DEBUG")'))


def fn_gdc() -> Dict[str, Any]:
    return {
        "enable_ocr": True,
        "camera": {"index": 1, "width": 1920, "height": 1080},
        "detection": {
            "h1_min": 0,
            "h1_max": 179,
            "s1_min": 0,
            "s1_max": 255,
            "v1_min": 0,
            "v1_max": 85,
            "h2_min": 0,
            "h2_max": 179,
            "s2_min": 0,
            "s2_max": 255,
            "v2_min": 0,
            "v2_max": 85,
            "use_range2": False,
            "min_area": 200,
            "canny_min": 50,
            "canny_max": 150,
        },
        "area_filter": {
            "min_crop_area": 500000,
            "max_crop_area": 19000000,
            "enable_area_filter": True,
            "a4_ratio_tolerance": 0.3,
            "max_circularity": 0.7,
            "min_solidity": 0.8,
            "max_vertices": 8,
            "enable_a4_check": True,
        },
        "perspective_correction": {
            "enable": True,
            "target_width": 210,
            "target_height": 297,
            "a4_ratio": 1.414285714285714,
            "use_short_edge_for_measurement": True,
        },
        "black_detection": {
            "lower_h": 0,
            "lower_s": 0,
            "lower_v": 0,
            "upper_h": 255,
            "upper_s": 255,
            "upper_v": 80,
            "morph_kernel_size": 3,
        },
    }


MIN_AREA: int = 10000
v_detpar: Dict[str, Union[int, float]] = {
    "min_area": 500,
    "hollow_ratio": 0.1,
    "aspect_ratio_min": 0.2,
    "aspect_ratio_max": 5.0,
    "epsilon_factor": 0.02,
    "min_frame_thickness": 10,
    "min_vertices": 4,
    "max_vertices": 5,
}
v_con_2: Dict[str, Any] = fn_loacon()
v_par: Dict[str, Any] = v_con_2["detection"].copy()
v_afp: Dict[str, Any] = v_con_2["area_filter"].copy()
v_perpar: Dict[str, Any] = v_con_2["perspective_correction"].copy()
v_bdp: Dict[str, Any] = v_con_2["black_detection"].copy()
v_camcon: Dict[str, Any] = v_con_2["camera"].copy()
if "min_vertices" in v_con_2["detection"]:
    v_detpar["min_vertices"] = v_con_2["detection"]["min_vertices"]
if "max_vertices" in v_con_2["detection"]:
    v_detpar["max_vertices"] = v_con_2["detection"]["max_vertices"]
v_cap: cv2.VideoCapture = cv2.VideoCapture(v_camcon["index"])
v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, v_camcon["width"])
v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, v_camcon["height"])
if not v_cap.isOpened():
    raise RuntimeError("无法打开摄像头")
v_lk: threading.Lock = threading.Lock()
v_frm: Optional[np.ndarray] = None
v_latsta: Dict[str, Any] = {}
v_curcro: List[np.ndarray] = []
v_cli: List[WebSocket] = []
v_sar: bool = False


def fn_dis(v_p: Tuple[float, float], v_q: Tuple[float, float]) -> float:
    return math.hypot(v_q[0] - v_p[0], v_q[1] - v_p[1])


def fn_mid(v_p: Tuple[int, int], v_q: Tuple[int, int]) -> Tuple[int, int]:
    return ((v_p[0] + v_q[0]) // 2, (v_p[1] + v_q[1]) // 2)


def fn_ordpts_2(v_pts_2: np.ndarray) -> np.ndarray:
    v_s = v_pts_2.sum(axis=1)
    v_dif = np.diff(v_pts_2, axis=1).ravel()
    v_tl, v_br = (v_pts_2[np.argmin(v_s)], v_pts_2[np.argmax(v_s)])
    v_tr, v_bl = (v_pts_2[np.argmin(v_dif)], v_pts_2[np.argmax(v_dif)])
    return np.array([v_tl, v_tr, v_br, v_bl], dtype=np.int32)


def fn_ia4q(v_pts_2: np.ndarray, v_a4r: float = 297 / 210, v_tol: float = 0.2) -> bool:
    v_o = fn_ordpts_2(v_pts_2)
    v_e = [fn_dis(v_o[v_i], v_o[(v_i + 1) % 4]) for v_i in range(4)]
    v_lone, v_shoe = ((v_e[0] + v_e[2]) / 2, (v_e[1] + v_e[3]) / 2)
    if v_shoe == 0:
        return False
    v_asp = v_lone / v_shoe
    return abs(v_asp - v_a4r) < v_tol or abs(v_asp - 1 / v_a4r) < v_tol


def fn_cpd(v_shpres: Dict[str, Any], v_crow: int, v_croh: int) -> Dict[str, Any]:
    v_pwm = 170
    v_phm = 257
    v_mppx = v_pwm / v_crow
    v_mppy = v_phm / v_croh
    v_mpp = (v_mppx + v_mppy) / 2
    v_shptyp = v_shpres["shape_type"]
    v_phyinf = {
        "mm_per_pixel": v_mpp,
        "physical_width_mm": 0,
        "physical_height_mm": 0,
        "physical_diameter_mm": 0,
        "physical_side_lengths_mm": [],
        "physical_perimeter_mm": 0,
        "physical_area_mm2": 0,
        "measurement_type": "unknown",
    }
    v_phyinf["physical_area_mm2"] = v_shpres["area"] * v_mpp**2
    v_phyinf["physical_perimeter_mm"] = v_shpres.get("perimeter", 0) * v_mpp
    if v_shptyp == "Square":
        if v_shpres.get("side_lengths"):
            v_psl = [v_len * v_mpp for v_len in v_shpres["side_lengths"]]
            v_phyinf["physical_side_lengths_mm"] = v_psl
            v_asl = sum(v_psl) / len(v_psl)
            v_phyinf["physical_width_mm"] = v_asl
            v_phyinf["physical_height_mm"] = v_asl
            v_phyinf["measurement_type"] = "side_length"
        else:
            v_phyinf["physical_width_mm"] = v_shpres["width"] * v_mpp
            v_phyinf["physical_height_mm"] = v_shpres["height"] * v_mpp
            v_phyinf["measurement_type"] = "bounding_box"
    elif v_shptyp == "Circle":
        v_diapxs = max(v_shpres["width"], v_shpres["height"])
        v_phyinf["physical_diameter_mm"] = v_diapxs * v_mpp
        v_phyinf["physical_width_mm"] = v_phyinf["physical_diameter_mm"]
        v_phyinf["physical_height_mm"] = v_phyinf["physical_diameter_mm"]
        v_phyinf["measurement_type"] = "diameter"
    elif v_shptyp == "Triangle":
        if v_shpres.get("side_lengths"):
            v_psl = [v_len * v_mpp for v_len in v_shpres["side_lengths"]]
            v_phyinf["physical_side_lengths_mm"] = v_psl
            v_asl = sum(v_psl) / len(v_psl)
            v_phyinf["physical_width_mm"] = v_asl
            v_phyinf["physical_height_mm"] = v_asl * 0.866
            v_phyinf["measurement_type"] = "equilateral_triangle"
        else:
            v_phyinf["physical_width_mm"] = v_shpres["width"] * v_mpp
            v_phyinf["physical_height_mm"] = v_shpres["height"] * v_mpp
            v_phyinf["measurement_type"] = "bounding_box"
    else:
        v_phyinf["physical_width_mm"] = v_shpres["width"] * v_mpp
        v_phyinf["physical_height_mm"] = v_shpres["height"] * v_mpp
        if v_shpres.get("side_lengths"):
            v_psl = [v_len * v_mpp for v_len in v_shpres["side_lengths"]]
            v_phyinf["physical_side_lengths_mm"] = v_psl
        v_phyinf["measurement_type"] = "bounding_box"
    return v_phyinf


def fn_isba(v_con_4: np.ndarray, v_arethr: int = 20) -> Dict[str, Any]:
    v_are = cv2.contourArea(v_con_4)
    if v_are < v_arethr:
        return {
            "shape_type": "Unknown",
            "area": v_are,
            "width": 0,
            "height": 0,
            "contour": v_con_4,
            "info": "Too small area",
            "side_lengths": [],
            "mean_side_length": 0,
            "perimeter": 0,
        }
    v_per_2 = cv2.arcLength(v_con_4, True)
    if v_per_2 == 0:
        return {
            "shape_type": "Unknown",
            "area": v_are,
            "width": 0,
            "height": 0,
            "contour": v_con_4,
            "info": "Zero perimeter",
            "side_lengths": [],
            "mean_side_length": 0,
            "perimeter": 0,
        }
    v_cir = 4 * np.pi * v_are / v_per_2**2
    v_x, v_y, v_w, v_h = cv2.boundingRect(v_con_4)
    v_eps = 0.02 * v_per_2
    v_app_2 = cv2.approxPolyDP(v_con_4, v_eps, True)
    v_ver = len(v_app_2)
    v_shptyp = "Unknown"
    v_sidlen_2 = []
    v_asl_2 = 0
    if v_ver >= 3:
        v_sidlen_2 = [
            float(np.linalg.norm(v_app_2[v_i][0] - v_app_2[(v_i + 1) % v_ver][0]))
            for v_i in range(v_ver)
        ]
        v_asl_2 = float(np.mean(v_sidlen_2))
    v_inf = f"Vertices={v_ver}, Circ={v_cir:.3f}, Ratio={v_w / v_h:.3f}"
    if v_ver == 3:
        v_shptyp = "Triangle"
        v_inf += f"; Triangle, Sides={[f'{v_s:.1f}' for v_s in v_sidlen_2]}"
    elif v_ver == 4:
        v_varcoe = np.std(v_sidlen_2) / v_asl_2 if v_asl_2 > 0 else 1
        v_inf += (
            f"; Var_coeff={v_varcoe:.3f}, Sides={[f'{v_s:.1f}' for v_s in v_sidlen_2]}"
        )
        if v_varcoe < 0.4:
            v_shptyp = "Square"
            v_w = v_h = int(v_asl_2)
            v_inf += "; Square"
        else:
            v_inf += "; Quad but not square"
    elif v_cir > 0.6:
        v_shptyp = "Circle"
        v_inf += "; Circle"
    elif v_ver >= 6 and v_cir > 0.4:
        v_shptyp = "Circle"
        v_inf += "; Near circle"
    else:
        v_inf += "; Unknown shape"
    return {
        "shape_type": v_shptyp,
        "area": v_are,
        "width": v_w
        if v_shptyp != "Circle"
        else int(cv2.minEnclosingCircle(v_con_4)[1] * 2),
        "height": v_h
        if v_shptyp != "Circle"
        else int(cv2.minEnclosingCircle(v_con_4)[1] * 2),
        "contour": v_con_4,
        "info": v_inf,
        "side_lengths": v_sidlen_2,
        "mean_side_length": v_asl_2,
        "perimeter": v_per_2,
    }


def fn_buimsk(v_hsv: np.ndarray, v_par: Dict[str, Any]) -> np.ndarray:
    v_low1 = np.array([v_par["h1_min"], v_par["s1_min"], v_par["v1_min"]])
    v_upp1 = np.array([v_par["h1_max"], v_par["s1_max"], v_par["v1_max"]])
    v_msk1 = cv2.inRange(v_hsv, v_low1, v_upp1)
    if v_par["use_range2"]:
        v_low2 = np.array([v_par["h2_min"], v_par["s2_min"], v_par["v2_min"]])
        v_upp2 = np.array([v_par["h2_max"], v_par["s2_max"], v_par["v2_max"]])
        v_msk2 = cv2.inRange(v_hsv, v_low2, v_upp2)
        v_msk = cv2.bitwise_or(v_msk1, v_msk2)
    else:
        v_msk = v_msk1
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    v_msk = cv2.morphologyEx(v_msk, cv2.MORPH_OPEN, v_ker)
    v_msk = cv2.morphologyEx(v_msk, cv2.MORPH_CLOSE, v_ker)
    return v_msk


def fn_fa4q(v_msk: np.ndarray, v_par: Dict[str, Any]) -> List[np.ndarray]:
    v_edg = cv2.Canny(v_msk, v_par["canny_min"], v_par["canny_max"])
    v_cnt_2, v__ = cv2.findContours(v_edg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_qua_2 = []
    for v_cnt in v_cnt_2:
        if cv2.contourArea(v_cnt) < MIN_AREA:
            continue
        v_per = cv2.arcLength(v_cnt, True)
        v_app_2 = cv2.approxPolyDP(v_cnt, 0.02 * v_per, True)
        if len(v_app_2) == 4 and fn_ia4q(v_app_2.reshape(4, 2)):
            v_qua_2.append(v_app_2.reshape(4, 2))
    return v_qua_2


def fn_preimg(v_frm: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    global v_par, v_detpar
    "图像预处理 - 使用HSV色彩空间"
    v_hsv = cv2.cvtColor(v_frm, cv2.COLOR_BGR2HSV)
    v_h, v_s, v_v = cv2.split(v_hsv)
    v_gry = cv2.cvtColor(v_frm, cv2.COLOR_BGR2GRAY)
    v_blu = cv2.GaussianBlur(v_gry, (5, 5), 0)
    v_thr = cv2.adaptiveThreshold(
        v_blu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    v_lowhsv = np.array([v_par["h1_min"], v_par["s1_min"], v_par["v1_min"]])
    v_upphsv = np.array([v_par["h1_max"], v_par["s1_max"], v_par["v1_max"]])
    v_hsvmsk = cv2.inRange(v_hsv, v_lowhsv, v_upphsv)
    v_com = cv2.bitwise_or(cv2.bitwise_not(v_thr), v_hsvmsk)
    v_ker = np.ones((1, 1), np.uint8)
    v_cle = cv2.morphologyEx(v_com, cv2.MORPH_CLOSE, v_ker)
    v_cle = cv2.morphologyEx(v_cle, cv2.MORPH_OPEN, v_ker)
    v_debinf = {
        "hsv": v_hsv,
        "h_channel": v_h,
        "s_channel": v_s,
        "v_channel": v_v,
        "gray": v_gry,
        "blurred": v_blu,
        "thresh": v_thr,
        "hsv_mask": v_hsvmsk,
        "combined": v_com,
        "hsv_params": v_par.copy(),
        "detection_params": v_detpar.copy(),
    }
    return (v_cle, v_debinf)


def fn_fhr(v_proimg: np.ndarray, v_debinf: Dict[str, Any]) -> List[Dict[str, Any]]:
    global v_detpar, v_par, v_afp
    "检测空心矩形框架"
    v_con_5, v__ = cv2.findContours(v_proimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    v_holrec = []
    v_allcon = []
    for v_con_4 in v_con_5:
        v_are = cv2.contourArea(v_con_4)
        v_allcon.append({"contour": v_con_4, "area": v_are, "is_valid": False})
        if v_are < v_detpar["min_area"]:
            continue
        v_eps = v_detpar["epsilon_factor"] * cv2.arcLength(v_con_4, True)
        v_app_2 = cv2.approxPolyDP(v_con_4, v_eps, True)
        if v_detpar["min_vertices"] <= len(v_app_2) <= v_detpar["max_vertices"]:
            v_x, v_y, v_w, v_h = cv2.boundingRect(v_app_2)
            if (
                v_w < v_detpar["min_frame_thickness"] * 3
                or v_h < v_detpar["min_frame_thickness"] * 3
            ):
                continue
            v_asprat = float(v_w) / v_h
            if v_detpar["aspect_ratio_min"] < v_asprat < v_detpar["aspect_ratio_max"]:
                if fn_ihr(v_proimg, v_con_4):
                    if v_afp["enable_area_filter"]:
                        if (
                            v_are < v_afp["min_crop_area"]
                            or v_are > v_afp["max_crop_area"]
                        ):
                            continue
                    v_holrec.append(
                        {
                            "contour": v_app_2,
                            "bbox": (v_x, v_y, v_w, v_h),
                            "area": v_are,
                            "aspect_ratio": v_asprat,
                        }
                    )
                    v_allcon[-1]["is_valid"] = True
    v_holrec.sort(key=lambda v_x: v_x["area"], reverse=True)
    v_debinf["all_contours"] = v_allcon
    v_debinf["valid_rectangles"] = len(v_holrec)
    return v_holrec


def fn_ihr(v_proimg: np.ndarray, v_con_4: np.ndarray) -> bool:
    global v_detpar, v_par
    "判断是否为空心矩形框架"
    v_x, v_y, v_w, v_h = cv2.boundingRect(v_con_4)
    v_msk = np.zeros(v_proimg.shape, np.uint8)
    cv2.fillPoly(v_msk, [v_con_4], 255)
    v_frmthi = v_detpar["min_frame_thickness"]
    v_innx = v_x + v_frmthi
    v_inny = v_y + v_frmthi
    v_innw = v_w - 2 * v_frmthi
    v_innh = v_h - 2 * v_frmthi
    if v_innw <= 0 or v_innh <= 0:
        return False
    v_innmsk = np.zeros(v_proimg.shape, np.uint8)
    cv2.rectangle(
        v_innmsk, (v_innx, v_inny), (v_innx + v_innw, v_inny + v_innh), 255, -1
    )
    v_commsk = cv2.bitwise_and(v_msk, v_innmsk)
    v_tip = cv2.countNonZero(v_commsk)
    if v_tip == 0:
        return False
    v_innreg = cv2.bitwise_and(v_proimg, v_commsk)
    v_whipxs = cv2.countNonZero(v_innreg)
    v_holrat = v_whipxs / v_tip if v_tip > 0 else 0
    v_frmmsk = cv2.bitwise_and(v_msk, cv2.bitwise_not(v_innmsk))
    v_frmreg = cv2.bitwise_and(v_proimg, v_frmmsk)
    v_frmtot = cv2.countNonZero(v_frmmsk)
    v_frmbla = v_frmtot - cv2.countNonZero(v_frmreg) if v_frmtot > 0 else 0
    v_frmrat = v_frmbla / v_frmtot if v_frmtot > 0 else 0
    v_ishol = v_holrat > v_detpar["hollow_ratio"]
    v_hasfrm = v_frmrat > 0.3
    return v_ishol and v_hasfrm


def fn_draann(v_frm: np.ndarray, v_holrec: List[Dict[str, Any]]) -> np.ndarray:
    global v_detpar, v_par
    "绘制标注"
    v_annfrm = v_frm.copy()
    for v_i, v_rec in enumerate(v_holrec):
        v_con_4 = v_rec["contour"]
        v_x, v_y, v_w, v_h = v_rec["bbox"]
        v_are = v_rec["area"]
        v_asprat = v_rec.get("aspect_ratio", 0)
        cv2.drawContours(v_annfrm, [v_con_4], -1, (0, 255, 0), 3)
        cv2.rectangle(v_annfrm, (v_x, v_y), (v_x + v_w, v_y + v_h), (255, 0, 0), 2)
        v_ctrx = v_x + v_w // 2
        v_ctry = v_y + v_h // 2
        cv2.circle(v_annfrm, (v_ctrx, v_ctry), 5, (0, 0, 255), -1)
        if len(v_con_4) >= 4:
            v_srcpts = v_con_4.reshape(-1, 2).astype(np.float32)
            if len(v_srcpts) > 4:
                v_hul = cv2.convexHull(v_srcpts.astype(np.int32))
                v_eps = 0.02 * cv2.arcLength(v_hul, True)
                v_app_2 = cv2.approxPolyDP(v_hul, v_eps, True)
                if len(v_app_2) >= 4:
                    v_srcpts = v_app_2[:4].reshape(-1, 2).astype(np.float32)
                else:
                    v_srcpts = v_srcpts[:4]
            v_ordpts_2 = fn_ordpts(v_srcpts)
            v_edg1, v_edg2, v_edg3, v_edg4 = fn_cel(v_srcpts)
            v_horavg = (v_edg1 + v_edg3) / 2
            v_veravg = (v_edg2 + v_edg4) / 2
            if v_horavg > v_veravg:
                v_lefmid = fn_mid(
                    (int(v_ordpts_2[0][0]), int(v_ordpts_2[0][1])),
                    (int(v_ordpts_2[3][0]), int(v_ordpts_2[3][1])),
                )
                v_rigmid = fn_mid(
                    (int(v_ordpts_2[1][0]), int(v_ordpts_2[1][1])),
                    (int(v_ordpts_2[2][0]), int(v_ordpts_2[2][1])),
                )
                v_newlen = fn_dis(v_lefmid, v_rigmid)
                v_mid1, v_mid2 = (v_lefmid, v_rigmid)
            else:
                v_topmid = fn_mid(
                    (int(v_ordpts_2[0][0]), int(v_ordpts_2[0][1])),
                    (int(v_ordpts_2[1][0]), int(v_ordpts_2[1][1])),
                )
                v_botmid = fn_mid(
                    (int(v_ordpts_2[3][0]), int(v_ordpts_2[3][1])),
                    (int(v_ordpts_2[2][0]), int(v_ordpts_2[2][1])),
                )
                v_newlen = fn_dis(v_topmid, v_botmid)
                v_mid1, v_mid2 = (v_topmid, v_botmid)
            cv2.line(v_annfrm, v_mid1, v_mid2, (0, 0, 255), 3)
            cv2.circle(v_annfrm, v_mid1, 6, (0, 0, 255), -1)
            cv2.circle(v_annfrm, v_mid2, 6, (0, 0, 255), -1)
            v_col = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            v_lab = ["TL", "TR", "BR", "BL"]
            for v_j, (v_pt_2, v_clr, v_lbl) in enumerate(zip(v_ordpts_2, v_col, v_lab)):
                cv2.circle(v_annfrm, (int(v_pt_2[0]), int(v_pt_2[1])), 8, v_clr, -1)
                cv2.putText(
                    v_annfrm,
                    v_lbl,
                    (int(v_pt_2[0]) + 10, int(v_pt_2[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    v_clr,
                    2,
                )
        else:
            v_newlen = 0
            v_horavg = v_veravg = 0
        v_lbl = f"Hollow Rect {v_i + 1}"
        v_inf = f"Area: {int(v_are)}"
        v_szinf = f"Size: {v_w}x{v_h}"
        v_ratinf = f"Ratio: {v_asprat:.2f}"
        v_edginf = f"H_avg:{v_horavg:.0f} V_avg:{v_veravg:.0f}"
        v_nli = f"Short Edge Length: {v_newlen:.0f}px"
        v_texlin = [v_lbl, v_inf, v_szinf, v_ratinf, v_edginf, v_nli]
        for v_idx, v_tex in enumerate(v_texlin):
            v_yoff = -70 + v_idx * 15
            cv2.putText(
                v_annfrm,
                v_tex,
                (v_x, v_y + v_yoff),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        for v_pt in v_con_4:
            v_px, v_py = v_pt[0]
            cv2.circle(v_annfrm, (v_px, v_py), 3, (255, 255, 0), -1)
    return v_annfrm


def fn_anninn(
    v_cro_2: List[np.ndarray], v_par: Dict[str, Any], v_sta: Dict[str, Any]
) -> None:
    v_h1l = np.array([v_par["h1_min"], v_par["s1_min"], v_par["v1_min"]])
    v_h1u = np.array([v_par["h1_max"], v_par["s1_max"], v_par["v1_max"]])
    v_h2l = (
        np.array([v_par["h2_min"], v_par["s2_min"], v_par["v2_min"]])
        if v_par["use_range2"]
        else None
    )
    v_h2u = (
        np.array([v_par["h2_max"], v_par["s2_max"], v_par["v2_max"]])
        if v_par["use_range2"]
        else None
    )
    for v_idx, v_roi in enumerate(v_cro_2):
        if v_idx >= len(v_sta["rects"]):
            continue
        v_gry = cv2.cvtColor(v_roi, cv2.COLOR_BGR2GRAY)
        v__, v_whi = cv2.threshold(v_gry, 100, 255, cv2.THRESH_BINARY)
        v_wcnt, v__ = cv2.findContours(
            v_whi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not v_wcnt:
            continue
        v_wx, v_wy, v_ww, v_wh = cv2.boundingRect(max(v_wcnt, key=cv2.contourArea))
        v_inn = v_roi[v_wy : v_wy + v_wh, v_wx : v_wx + v_ww]
        v_hsv = cv2.cvtColor(v_inn, cv2.COLOR_BGR2HSV)
        v_msk = cv2.inRange(v_hsv, v_h1l, v_h1u)
        if v_par["use_range2"] and v_h2l is not None and (v_h2u is not None):
            v_msk2 = cv2.inRange(v_hsv, v_h2l, v_h2u)
            v_msk = cv2.bitwise_or(v_msk, v_msk2)
        v_k = np.ones((3, 3), np.uint8)
        v_msk = cv2.morphologyEx(v_msk, cv2.MORPH_OPEN, v_k)
        v_msk = cv2.morphologyEx(v_msk, cv2.MORPH_CLOSE, v_k)
        v_scnt, v__ = cv2.findContours(
            v_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not v_scnt:
            continue
        v_detsha = []
        for v_shp in v_scnt:
            v_shpare = cv2.contourArea(v_shp)
            if v_shpare < 50:
                continue
            v_shpres = fn_isba(v_shp)
            v_detsha.append(v_shpres)
        if not v_detsha:
            continue
        v_asi = []
        v_tia = 0
        for v_shpres in v_detsha:
            v_tia += v_shpres["area"]
            v_phyinf = fn_cpd(v_shpres, v_roi.shape[1], v_roi.shape[0])
            v_shp = v_shpres["contour"]
            M = cv2.moments(v_shp)
            if M["m00"] != 0:
                v_cx_2 = int(M["m10"] / M["m00"]) + v_wx
                v_cy_2 = int(M["m01"] / M["m00"]) + v_wy
                v_ctr = [v_cx_2, v_cy_2]
            else:
                v_x, v_y, v_w, v_h = cv2.boundingRect(v_shp)
                v_ctr = [v_x + v_w // 2 + v_wx, v_y + v_h // 2 + v_wy]
            v_x, v_y, v_w, v_h = cv2.boundingRect(v_shp)
            v_bbox = [v_x + v_wx, v_y + v_wy, v_w, v_h]
            v_shpinf = {
                "shape_type": v_shpres["shape_type"],
                "width": int(v_shpres["width"]),
                "height": int(v_shpres["height"]),
                "area": int(v_shpres["area"]),
                "info": v_shpres["info"],
                "contour": v_shpres["contour"].reshape(-1, 2).tolist(),
                "side_lengths": [
                    float(v_len) for v_len in v_shpres.get("side_lengths", [])
                ],
                "mean_side_length": float(v_shpres.get("mean_side_length", 0)),
                "perimeter": float(v_shpres.get("perimeter", 0)),
                "physical_info": v_phyinf,
                "position": {
                    "center": v_ctr,
                    "bbox": v_bbox,
                    "contour_points": v_shpres["contour"].reshape(-1, 2).tolist(),
                },
            }
            v_asi.append(v_shpinf)
        v_maishp = max(v_detsha, key=lambda v_x: v_x["area"])
        v_mpi = fn_cpd(v_maishp, v_roi.shape[1], v_roi.shape[0])
        v_sta["rects"][v_idx].update(
            {
                "shape_type": v_maishp["shape_type"],
                "inner_width": int(v_maishp["width"]),
                "inner_height": int(v_maishp["height"]),
                "inner_area": int(v_maishp["area"]),
                "inner_info": v_maishp["info"],
                "inner_contour": v_maishp["contour"].reshape(-1, 2).tolist(),
                "inner_side_lengths": [
                    float(v_len) for v_len in v_maishp.get("side_lengths", [])
                ],
                "inner_mean_side_length": float(v_maishp.get("mean_side_length", 0)),
                "inner_perimeter": float(v_maishp.get("perimeter", 0)),
                "inner_physical_info": v_mpi,
                "all_shapes": v_asi,
                "shapes_count": len(v_detsha),
                "total_inner_area": int(v_tia),
            }
        )
        for v_i, v_shpres in enumerate(v_detsha):
            v_shp = v_shpres["contour"]
            v_col = [
                (0, 255, 0),
                (255, 0, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
            ]
            v_clr = v_col[v_i % len(v_col)]
            cv2.drawContours(v_roi, [v_shp], -1, v_clr, 2)
            M = cv2.moments(v_shp)
            if M["m00"] != 0:
                v_cx_2 = int(M["m10"] / M["m00"]) + v_wx
                v_cy_2 = int(M["m01"] / M["m00"]) + v_wy
                cv2.circle(v_roi, (v_cx_2, v_cy_2), 5, v_clr, -1)
                cv2.putText(
                    v_roi,
                    f"{v_shpres['shape_type']} #{v_i + 1}",
                    (v_cx_2 - 40, v_cy_2 - 30 - v_i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    v_clr,
                    2,
                )
                cv2.putText(
                    v_roi,
                    f"Area: {int(v_shpres['area'])}px",
                    (v_cx_2 - 40, v_cy_2 - 10 - v_i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    v_clr,
                    1,
                )
                v_phyinf = fn_cpd(v_shpres, v_roi.shape[1], v_roi.shape[0])
                if v_shpres["shape_type"] == "Square":
                    if v_phyinf["physical_side_lengths_mm"]:
                        v_avgsid = sum(v_phyinf["physical_side_lengths_mm"]) / len(
                            v_phyinf["physical_side_lengths_mm"]
                        )
                        cv2.putText(
                            v_roi,
                            f"Side: {v_avgsid:.1f}mm",
                            (v_cx_2 - 40, v_cy_2 + 10 - v_i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            v_clr,
                            1,
                        )
                elif v_shpres["shape_type"] == "Circle":
                    cv2.putText(
                        v_roi,
                        f"Dia: {v_phyinf['physical_diameter_mm']:.1f}mm",
                        (v_cx_2 - 40, v_cy_2 + 10 - v_i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        v_clr,
                        1,
                    )
                elif v_shpres["shape_type"] == "Triangle":
                    if v_phyinf["physical_side_lengths_mm"]:
                        v_avgsid = sum(v_phyinf["physical_side_lengths_mm"]) / len(
                            v_phyinf["physical_side_lengths_mm"]
                        )
                        cv2.putText(
                            v_roi,
                            f"Side: {v_avgsid:.1f}mm (eq.tri)",
                            (v_cx_2 - 40, v_cy_2 + 10 - v_i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            v_clr,
                            1,
                        )
                else:
                    cv2.putText(
                        v_roi,
                        f"Size: {v_phyinf['physical_width_mm']:.1f}x{v_phyinf['physical_height_mm']:.1f}mm",
                        (v_cx_2 - 40, v_cy_2 + 10 - v_i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        v_clr,
                        1,
                    )
                if v_shpres.get("side_lengths") and len(v_shpres["side_lengths"]) > 0:
                    v_sidlen_2 = v_shpres["side_lengths"]
                    if len(v_sidlen_2) == 4:
                        v_sidtex = f"Px: {v_sidlen_2[0]:.0f},{v_sidlen_2[1]:.0f},{v_sidlen_2[2]:.0f},{v_sidlen_2[3]:.0f}"
                    elif len(v_sidlen_2) == 3:
                        v_sidtex = f"Px: {v_sidlen_2[0]:.0f},{v_sidlen_2[1]:.0f},{v_sidlen_2[2]:.0f}"
                    else:
                        v_sidtex = f"Px: {v_shpres.get('mean_side_length', 0):.0f}"
                    cv2.putText(
                        v_roi,
                        v_sidtex,
                        (v_cx_2 - 40, v_cy_2 + 30 - v_i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        v_clr,
                        1,
                    )


def fn_ordpts(v_pts_2: np.ndarray) -> np.ndarray:
    v_ctr = np.mean(v_pts_2, axis=0)
    v_ang = np.arctan2(v_pts_2[:, 1] - v_ctr[1], v_pts_2[:, 0] - v_ctr[0])
    v_soridxs = np.argsort(v_ang)
    v_sorpts = v_pts_2[v_soridxs]
    v_topidxs = np.argsort(v_sorpts[:, 1])[:2]
    v_toppts = v_sorpts[v_topidxs]
    v_botidxs = np.argsort(v_sorpts[:, 1])[-2:]
    v_botpts = v_sorpts[v_botidxs]
    if v_toppts[0][0] > v_toppts[1][0]:
        v_toppts = v_toppts[::-1]
    if v_botpts[0][0] > v_botpts[1][0]:
        v_botpts = v_botpts[::-1]
    return np.array(
        [v_toppts[0], v_toppts[1], v_botpts[1], v_botpts[0]], dtype=np.float32
    )


def fn_cel(v_pts_2: np.ndarray) -> Tuple[float, float, float, float]:
    v_ordpts_2 = fn_ordpts(v_pts_2)
    v_edg1 = np.linalg.norm(v_ordpts_2[1] - v_ordpts_2[0])
    v_edg2 = np.linalg.norm(v_ordpts_2[2] - v_ordpts_2[1])
    v_edg3 = np.linalg.norm(v_ordpts_2[3] - v_ordpts_2[2])
    v_edg4 = np.linalg.norm(v_ordpts_2[0] - v_ordpts_2[3])
    return (v_edg1, v_edg2, v_edg3, v_edg4)


def fn_ca4cfc(v_img_2: np.ndarray, v_holrec: List[Dict[str, Any]]) -> List[np.ndarray]:
    v_cro_2 = []
    v_twm = 170
    v_thm = 257
    v_a4r = v_thm / v_twm
    for v_rec in v_holrec:
        v_conpts = v_rec["contour"]
        if len(v_conpts) < 4:
            continue
        if len(v_conpts) > 4:
            v_hul = cv2.convexHull(v_conpts)
            v_eps = 0.02 * cv2.arcLength(v_hul, True)
            v_app_2 = cv2.approxPolyDP(v_hul, v_eps, True)
            if len(v_app_2) >= 4:
                v_conpts = v_app_2[:4]
            else:
                v_conpts = v_conpts[:4]
        v_srcpts = v_conpts.reshape(-1, 2).astype(np.float32)
        v_ordsrc = fn_ordpts(v_srcpts)
        v_topw = np.linalg.norm(v_ordsrc[1] - v_ordsrc[0])
        v_botw = np.linalg.norm(v_ordsrc[3] - v_ordsrc[2])
        v_lefh = np.linalg.norm(v_ordsrc[0] - v_ordsrc[3])
        v_righ = np.linalg.norm(v_ordsrc[2] - v_ordsrc[1])
        v_detw = (v_topw + v_botw) / 2
        v_deth = (v_lefh + v_righ) / 2
        if v_detw >= v_deth:
            v_tarw = int(v_detw)
            v_tarh = int(v_detw * v_a4r)
        else:
            v_tarh = int(v_deth)
            v_tarw = int(v_deth / v_a4r)
        v_dstpts_2 = np.array(
            [[0, 0], [v_tarw - 1, 0], [v_tarw - 1, v_tarh - 1], [0, v_tarh - 1]],
            dtype=np.float32,
        )
        try:
            v_mat = cv2.getPerspectiveTransform(v_ordsrc, v_dstpts_2)
            v_war = cv2.warpPerspective(v_img_2, v_mat, (v_tarw, v_tarh))
            v_cro_2.append(v_war)
        except Exception as v_e:
            print(eval('str("DEBUG")'))
            continue
    return v_cro_2


def fn_daa_2(
    v_annfrm: np.ndarray,
    v_holrec: List[Dict[str, Any]],
    v_cro_2: List[np.ndarray],
    v_debinf: Dict[str, Any],
) -> None:
    for v_i, v_rec in enumerate(v_holrec):
        v_con_4 = v_rec["contour"]
        if len(v_con_4) >= 4:
            v_srcpts = v_con_4.reshape(-1, 2).astype(np.float32)
            if len(v_srcpts) > 4:
                v_hul = cv2.convexHull(v_srcpts.astype(np.int32))
                v_eps = 0.02 * cv2.arcLength(v_hul, True)
                v_app_2 = cv2.approxPolyDP(v_hul, v_eps, True)
                if len(v_app_2) >= 4:
                    v_srcpts = v_app_2[:4].reshape(-1, 2).astype(np.float32)
                else:
                    v_srcpts = v_srcpts[:4]
            v_edg1, v_edg2, v_edg3, v_edg4 = fn_cel(v_srcpts)
            v_horavg = (v_edg1 + v_edg3) / 2
            v_veravg = (v_edg2 + v_edg4) / 2
            v_ordpts_2 = fn_ordpts(v_srcpts)
            v_col = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            v_lab = ["TL", "TR", "BR", "BL"]
            for v_j, (v_pt_2, v_clr, v_lbl) in enumerate(zip(v_ordpts_2, v_col, v_lab)):
                cv2.circle(v_annfrm, (int(v_pt_2[0]), int(v_pt_2[1])), 8, v_clr, -1)
                cv2.putText(
                    v_annfrm,
                    v_lbl,
                    (int(v_pt_2[0]) + 10, int(v_pt_2[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    v_clr,
                    2,
                )
            v_x, v_y, v_w, v_h = v_rec["bbox"]
            v_edginf = f"H_avg:{v_horavg:.0f} V_avg:{v_veravg:.0f}"
            cv2.putText(
                v_annfrm,
                v_edginf,
                (v_x, v_y + v_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
    v_inftex = f"Detected: {len(v_holrec)} Crops: {len(v_cro_2)}"
    cv2.putText(
        v_annfrm, v_inftex, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    v_debtex = f"Total Contours: {len(v_debinf.get('all_contours', []))}"
    cv2.putText(
        v_annfrm, v_debtex, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )


def fn_daa(v_img_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    global v_watimg, v_curcro, v_rawcro
    v_sta = {
        "count": 0,
        "rects": [],
        "area_sum": 0,
        "inner_rectangles": [],
        "inner_count": 0,
        "crops_count": 0,
        "black_squares": [],
        "minimum_black_square": {"found": False},
    }
    v_pro_2, v_debinf = fn_preimg(v_img_2)
    v_msk = v_pro_2.copy()
    v_holrec = fn_fhr(v_pro_2, v_debinf)
    v_cro_2 = fn_ca4cfc(v_img_2, v_holrec)
    v_curcro = v_cro_2
    v_rawcro = [v_cro.copy() for v_cro in v_cro_2]
    v_annfrm = fn_draann(v_img_2, v_holrec)
    fn_daa_2(v_annfrm, v_holrec, v_cro_2, v_debinf)
    v_sta["count"] = len(v_holrec)
    v_sta["inner_count"] = len(v_holrec)
    v_sta["crops_count"] = len(v_cro_2)
    for v_i, v_rec in enumerate(v_holrec):
        v_x, v_y, v_w, v_h = v_rec["bbox"]
        v_are = v_rec["area"]
        v_conpts = v_rec["contour"].reshape(-1, 2).astype(np.float32)
        if len(v_conpts) >= 4:
            v_edg1, v_edg2, v_edg3, v_edg4 = fn_cel(v_conpts[:4])
            v_horavg = (v_edg1 + v_edg3) / 2
            v_veravg = (v_edg2 + v_edg4) / 2
            v_ordpts_2 = fn_ordpts(v_conpts[:4])
            if v_horavg > v_veravg:
                v_lefmid = (
                    (v_ordpts_2[0][0] + v_ordpts_2[3][0]) / 2,
                    (v_ordpts_2[0][1] + v_ordpts_2[3][1]) / 2,
                )
                v_rigmid = (
                    (v_ordpts_2[1][0] + v_ordpts_2[2][0]) / 2,
                    (v_ordpts_2[1][1] + v_ordpts_2[2][1]) / 2,
                )
                v_newlen = fn_dis(v_lefmid, v_rigmid)
            else:
                v_topmid = (
                    (v_ordpts_2[0][0] + v_ordpts_2[1][0]) / 2,
                    (v_ordpts_2[0][1] + v_ordpts_2[1][1]) / 2,
                )
                v_botmid = (
                    (v_ordpts_2[3][0] + v_ordpts_2[2][0]) / 2,
                    (v_ordpts_2[3][1] + v_ordpts_2[2][1]) / 2,
                )
                v_newlen = fn_dis(v_topmid, v_botmid)
        else:
            v_horavg = v_veravg = v_newlen = 0
        v_recinf = {
            "id": v_i + 1,
            "outer_width": int(v_w),
            "outer_height": int(v_h),
            "area": int(v_are),
            "position": (int(v_x), int(v_y)),
            "aspect_ratio": float(v_rec.get("aspect_ratio", 0)),
            "horizontal_avg": float(v_horavg),
            "vertical_avg": float(v_veravg),
            "new_long_px": float(v_newlen),
            "crop_width": v_cro_2[v_i].shape[1] if v_i < len(v_cro_2) else 0,
            "crop_height": v_cro_2[v_i].shape[0] if v_i < len(v_cro_2) else 0,
        }
        v_sta["rects"].append(v_recinf)
        v_inninf = {
            "id": v_i + 1,
            "bbox": (v_x, v_y, v_w, v_h),
            "area": int(v_are),
            "aspect_ratio": float(v_rec.get("aspect_ratio", 0)),
            "center": [v_x + v_w // 2, v_y + v_h // 2],
            "width": int(v_w),
            "height": int(v_h),
            "horizontal_avg": float(v_horavg),
            "vertical_avg": float(v_veravg),
            "new_long_px": float(v_newlen),
            "crop_generated": v_i < len(v_cro_2),
        }
        v_sta["inner_rectangles"].append(v_inninf)
    fn_anninn(v_cro_2, v_par, v_sta)
    v_neocro = [v_cro.copy() for v_cro in v_rawcro]
    fn_mabc(v_neocro, v_sta, v_img_2)
    return (v_annfrm, v_msk, v_sta)


def fn_jct(v_binimg: np.ndarray, v_cor: Tuple[int, int], v_rad: int = 30) -> bool:
    v_h, v_w = v_binimg.shape
    v_x, v_y = v_cor
    v_mskcirc = np.zeros((v_h, v_w), dtype=np.uint8)
    cv2.circle(v_mskcirc, (v_x, v_y), v_rad, 255, -1)
    v_roi = v_binimg & v_mskcirc
    v_totpxs = cv2.countNonZero(v_mskcirc)
    v_whipxs = cv2.countNonZero(v_roi)
    v_blapxs = v_totpxs - v_whipxs
    return v_whipxs < v_blapxs


def fn_caldis(v_pt1: Tuple[int, int], v_pt2: Tuple[int, int]) -> float:
    return np.sqrt((v_pt1[0] - v_pt2[0]) ** 2 + (v_pt1[1] - v_pt2[1]) ** 2)


def fn_pcfms(
    v_croimg: np.ndarray, v_mat_2: int = 500, v_pwm: float = 170.0, v_phm: float = 257.0
) -> Dict[str, Any]:
    if v_croimg is None or v_croimg.size == 0:
        return {
            "success": False,
            "error": "输入图像为空",
            "shortest_edge_length_px": 0,
            "shortest_edge_length_mm": 0,
            "component_id": -1,
            "start_point": None,
            "end_point": None,
            "all_valid_edges": [],
            "annotated_image": None,
        }
    v_gry = cv2.cvtColor(v_croimg, cv2.COLOR_BGR2GRAY)
    v__, v_bininv = cv2.threshold(v_gry, int(v_gry.mean()), 255, cv2.THRESH_BINARY_INV)
    v_h, v_w = v_bininv.shape
    for v_see in [(0, 0), (v_w - 20, 0), (0, v_h - 20), (v_w - 20, v_h - 20)]:
        if 0 <= v_see[0] < v_w and 0 <= v_see[1] < v_h:
            cv2.floodFill(v_bininv, None, v_see, 0)
    v_numlab, v_lab, v_sta, v__ = cv2.connectedComponentsWithStats(v_bininv)
    v_filbin = np.zeros_like(v_bininv)
    for v_lblid in range(1, v_numlab):
        v_are = v_sta[v_lblid, cv2.CC_STAT_AREA]
        if v_are >= v_mat_2:
            v_filbin[v_lab == v_lblid] = 255
    v_croh, v_crow = v_croimg.shape[:2]
    v_mppx = v_pwm / v_crow
    v_mppy = v_phm / v_croh
    v_mpp = (v_mppx + v_mppy) / 2
    v_nlf, v_labfil, v_stafil, v__ = cv2.connectedComponentsWithStats(v_filbin)
    if v_nlf < 2:
        return {
            "success": False,
            "error": f"过滤后未检测到有效连通域（面积均小于 {v_mat_2} 像素）",
            "shortest_edge_length_px": 0,
            "shortest_edge_length_mm": 0,
            "component_id": -1,
            "start_point": None,
            "end_point": None,
            "all_valid_edges": [],
            "annotated_image": v_croimg.copy(),
        }
    v_resimg = v_croimg.copy()
    v_valedg = []
    for v_lblid in range(1, v_nlf):
        v_are = v_stafil[v_lblid, cv2.CC_STAT_AREA]
        v_msk = (v_labfil == v_lblid).astype(np.uint8) * 255
        v_con_5, v__ = cv2.findContours(
            v_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not v_con_5:
            continue
        v_cnt = max(v_con_5, key=cv2.contourArea)
        v_per = cv2.arcLength(v_cnt, True)
        v_eps = 0.01 * v_per
        v_app_2 = cv2.approxPolyDP(v_cnt, v_eps, True).reshape(-1, 2)
        v_cor_2 = [(int(v_x), int(v_y)) for v_x, v_y in v_app_2]
        if len(v_cor_2) < 3:
            continue
        v_cortyp = [fn_jct(v_filbin, v_cor) for v_cor in v_cor_2]
        v_numcor = len(v_cor_2)
        for v_i in range(v_numcor):
            v_j = (v_i + 1) % v_numcor
            v_pt1 = v_cor_2[v_i]
            v_pt2 = v_cor_2[v_j]
            v_typ1 = v_cortyp[v_i]
            v_typ2 = v_cortyp[v_j]
            cv2.line(v_resimg, v_pt1, v_pt2, (0, 255, 0), 2)
            if v_typ1 and v_typ2:
                v_len = fn_caldis(v_pt1, v_pt2)
                v_valedg.append((v_pt1, v_pt2, v_len, v_lblid))
    if not v_valedg:
        return {
            "success": False,
            "error": "未检测到有效线段（两端均为外角点的线段）",
            "shortest_edge_length_px": 0,
            "shortest_edge_length_mm": 0,
            "component_id": -1,
            "start_point": None,
            "end_point": None,
            "all_valid_edges": [],
            "annotated_image": v_resimg,
        }
    v_shoedg = min(v_valedg, key=lambda v_x: v_x[2])
    v_pt1, v_pt2, v_sholen, v_comid = v_shoedg
    cv2.line(v_resimg, v_pt1, v_pt2, (0, 0, 255), 3)
    v_midpt = ((v_pt1[0] + v_pt2[0]) // 2, (v_pt1[1] + v_pt2[1]) // 2)
    cv2.putText(
        v_resimg,
        f"MIN：{v_sholen:.1f}px",
        v_midpt,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    v_diamm = v_sholen * v_mpp
    return {
        "success": True,
        "shortest_edge_length_px": v_sholen,
        "shortest_edge_length_mm": v_diamm,
        "component_id": v_comid,
        "start_point": v_pt1,
        "end_point": v_pt2,
        "all_valid_edges": v_valedg,
        "annotated_image": v_resimg,
        "mm_per_pixel": v_mpp,
        "filtered_components": v_nlf - 1,
    }


def fn_pmc(
    v_cro_2: List[np.ndarray],
    v_mat_2: int = 10,
    v_pwm: float = 170.0,
    v_phm: float = 257.0,
) -> List[Dict[str, Any]]:
    v_res = []
    for v_i, v_cro in enumerate(v_cro_2):
        v_res_2 = fn_pcfms(v_cro, v_mat_2=v_mat_2, v_pwm=v_pwm, v_phm=v_phm)
        v_res_2["crop_index"] = v_i
        v_res.append(v_res_2)
        if v_res_2["success"]:
            pass
        else:
            print(eval('str("DEBUG")'))
    return v_res


def fn_fgms(v_res: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    v_sucres = [v_r for v_r in v_res if v_r["success"]]
    if not v_sucres:
        return None
    v_mnres = min(v_sucres, key=lambda v_x: v_x["shortest_edge_length_px"])
    v_mnres["is_global_minimum"] = True
    return v_mnres


def fn_cdrtlf(
    v_res_2: Dict[str, Any], v_croctr: Tuple[float, float] = (0, 0)
) -> Optional[Dict[str, Any]]:
    if not v_res_2["success"]:
        return None
    v_stapt = v_res_2["start_point"]
    v_endpt = v_res_2["end_point"]
    v_ctrx = (v_stapt[0] + v_endpt[0]) / 2 + v_croctr[0]
    v_ctry = (v_stapt[1] + v_endpt[1]) / 2 + v_croctr[1]
    v_slp = v_res_2["shortest_edge_length_px"]
    v_slm = v_res_2["shortest_edge_length_mm"]
    v_arepx = v_slp**2
    v_halsid = v_slp / 2
    v_box = np.array(
        [
            [v_ctrx - v_halsid, v_ctry - v_halsid],
            [v_ctrx + v_halsid, v_ctry - v_halsid],
            [v_ctrx + v_halsid, v_ctry + v_halsid],
            [v_ctrx - v_halsid, v_ctry + v_halsid],
        ],
        dtype=np.int32,
    )
    return {
        "center": (v_ctrx, v_ctry),
        "area": v_arepx,
        "side_length": v_slp,
        "side_length_mm": v_slm,
        "aspect_ratio": 1.0,
        "type": "minimum_square_detected",
        "box": v_box,
        "is_minimum": True,
    }


def fn_mabc(
    v_cro_2: List[np.ndarray],
    v_sta: Dict[str, Any],
    v_img_2: np.ndarray,
    v_mat_2: int = 500,
    v_pwm: float = 170.0,
    v_phm: float = 257.0,
) -> None:
    global v_msi
    v_msi = []
    v_allres = fn_pmc(v_cro_2, v_mat_2=v_mat_2, v_pwm=v_pwm, v_phm=v_phm)
    v_gmr = fn_fgms(v_allres)
    for v_res_2 in v_allres:
        if v_res_2["success"]:
            v_msi.append(v_res_2["annotated_image"])
        else:
            v_croidx = v_res_2.get("crop_index", 0)
            if v_croidx < len(v_cro_2):
                v_msi.append(v_cro_2[v_croidx].copy())
            else:
                v_msi.append(np.zeros((100, 100, 3), dtype=np.uint8))
    if v_gmr and v_gmr["success"]:
        v_elp = v_gmr["shortest_edge_length_px"]
        v_elm = v_gmr["shortest_edge_length_mm"]
        v_stapt = v_gmr["start_point"]
        v_endpt = v_gmr["end_point"]
        v_ctrx = (v_stapt[0] + v_endpt[0]) / 2
        v_ctry = (v_stapt[1] + v_endpt[1]) / 2
        fn_dmec(v_img_2, v_ctrx, v_ctry, v_elp, v_elm)
        v_sta["minimum_black_square"] = {
            "found": True,
            "center": [float(v_ctrx), float(v_ctry)],
            "edge_length_px": float(v_elp),
            "edge_length_mm": float(v_elm),
            "start_point": [float(v_stapt[0]), float(v_stapt[1])],
            "end_point": [float(v_endpt[0]), float(v_endpt[1])],
            "type": "minimum_edge_detected",
        }
        v_sta["black_squares"] = [
            {
                "center": [float(v_ctrx), float(v_ctry)],
                "edge_length_px": float(v_elp),
                "edge_length_mm": float(v_elm),
                "is_minimum": True,
                "type": "minimum_edge_detected",
            }
        ]
    else:
        v_sta["minimum_black_square"] = {"found": False}
        v_sta["black_squares"] = []


def fn_dmec(
    v_img_2: np.ndarray, v_ctrx: float, v_ctry: float, v_elp: float, v_elm: float
) -> None:
    cv2.circle(v_img_2, (int(v_ctrx), int(v_ctry)), 12, (0, 255, 255), -1)
    cv2.putText(
        v_img_2,
        f"MIN EDGE:{int(v_elp)}px",
        (int(v_ctrx) - 60, int(v_ctry) - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        v_img_2,
        f"{v_elm:.1f}mm",
        (int(v_ctrx) - 30, int(v_ctry) + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )


def fn_dmsc(v_img_2: np.ndarray, v_mnsqu: Dict[str, Any]) -> None:
    if v_mnsqu is None:
        return
    v_ctr = v_mnsqu["center"]
    v_box = v_mnsqu.get("box")
    if v_box is not None:
        cv2.drawContours(v_img_2, [v_box], -1, (255, 255, 0), 4)
    cv2.circle(v_img_2, (int(v_ctr[0]), int(v_ctr[1])), 12, (0, 255, 255), -1)
    v_sidlen = v_mnsqu.get("side_length", 0)
    v_slm = v_mnsqu.get("side_length_mm", 0)
    cv2.putText(
        v_img_2,
        f"MIN EDGE:{int(v_sidlen)}px",
        (int(v_ctr[0]) - 60, int(v_ctr[1]) - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        v_img_2,
        f"{v_slm:.1f}mm",
        (int(v_ctr[0]) - 30, int(v_ctr[1]) + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )


def fn_mab(
    v_cro_2: List[np.ndarray], v_sta: Dict[str, Any], v_img_2: np.ndarray
) -> None:
    global v_msi
    v_msi = []
    for v_idx, v_cro in enumerate(v_cro_2):
        v_abs, v_blamsk, v_watimg = fn_dbsh(v_cro)
        v_minsqu = fn_fms(v_abs)
        fn_dms(v_img_2, v_minsqu)
        v_msi.append(v_watimg.copy() if v_watimg is not None else v_cro.copy())
        if v_minsqu:
            v_sta["minimum_black_square"] = {
                "found": True,
                "center": [float(v_minsqu["center"][0]), float(v_minsqu["center"][1])],
                "area": float(v_minsqu["area"]),
                "side_length": float(v_minsqu["side_length"]),
                "aspect_ratio": float(v_minsqu["aspect_ratio"]),
                "type": v_minsqu["type"],
            }
            v_sta["black_squares"] = [
                {
                    "center": [
                        float(v_minsqu["center"][0]),
                        float(v_minsqu["center"][1]),
                    ],
                    "area": float(v_minsqu["area"]),
                    "side_length": float(v_minsqu["side_length"]),
                    "aspect_ratio": float(v_minsqu["aspect_ratio"]),
                    "is_minimum": True,
                }
            ]
        else:
            v_sta["minimum_black_square"] = {"found": False}
            v_sta["black_squares"] = []


def fn_dbsh(
    v_img: np.ndarray, v_debimg: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    if v_debimg is None:
        v_debimg = v_img.copy()
    v_hsv = cv2.cvtColor(v_img, cv2.COLOR_BGR2HSV)
    v_lowbla = np.array([0, 0, 0])
    v_uppbla = np.array([180, 255, 80])
    v_msk = cv2.inRange(v_hsv, v_lowbla, v_uppbla)
    v_ker = np.ones((5, 5), np.uint8)
    v_msk = cv2.morphologyEx(v_msk, cv2.MORPH_CLOSE, v_ker)
    v_msk = cv2.morphologyEx(v_msk, cv2.MORPH_OPEN, v_ker)
    v_sepsqu, v_watdeb = fn_soss(v_msk, v_debimg)
    return (v_sepsqu, v_msk, v_watdeb)


def fn_soss(
    v_msk: np.ndarray, v_debimg: np.ndarray
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    v_distra = cv2.distanceTransform(v_msk, cv2.DIST_L2, 5)
    v__, v_surfg = cv2.threshold(v_distra, 0.4 * v_distra.max(), 255, 0)
    v_surfg = np.uint8(v_surfg)
    v__, v_mar = cv2.connectedComponents(v_surfg)
    v_watinp = cv2.cvtColor(v_msk, cv2.COLOR_GRAY2BGR)
    v_mar = cv2.watershed(v_watinp, v_mar)
    v_watdeb = np.zeros_like(v_debimg)
    v_watdeb[v_mar == -1] = [0, 0, 255]
    v_col = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for v_i in range(1, v_mar.max() + 1):
        v_clr = v_col[(v_i - 1) % len(v_col)]
        v_watdeb[v_mar == v_i] = v_clr
    v_blasqu = []
    for v_marid in range(1, v_mar.max() + 1):
        v_regmsk = (v_mar == v_marid).astype(np.uint8) * 255
        v_con_5, v__ = cv2.findContours(
            v_regmsk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for v_con_4 in v_con_5:
            v_are = cv2.contourArea(v_con_4)
            if v_are < 400:
                continue
            v_rec = cv2.minAreaRect(v_con_4)
            v_box = cv2.boxPoints(v_rec)
            v_box = np.int32(v_box)
            v_w_2, v_h_2 = v_rec[1]
            if v_w_2 > 0 and v_h_2 > 0:
                v_asprat = max(v_w_2, v_h_2) / min(v_w_2, v_h_2)
                if v_asprat < 2.5:
                    v_blasqu.append(
                        {
                            "type": "black_square",
                            "contour": v_con_4,
                            "box": v_box,
                            "area": v_are,
                            "aspect_ratio": v_asprat,
                            "center": v_rec[0],
                            "side_length": math.sqrt(v_are),
                        }
                    )
                    cv2.drawContours(v_debimg, [v_box], -1, (255, 0, 255), 3)
                    cv2.circle(
                        v_debimg,
                        (int(v_rec[0][0]), int(v_rec[0][1])),
                        8,
                        (255, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        v_debimg,
                        f"SEP-{v_marid}:{int(math.sqrt(v_are))}",
                        (int(v_rec[0][0]) - 40, int(v_rec[0][1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        2,
                    )
    if len(v_blasqu) == 0:
        v_con_5, v__ = cv2.findContours(
            v_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for v_con_4 in v_con_5:
            v_are = cv2.contourArea(v_con_4)
            if v_are < 400:
                continue
            v_rec = cv2.minAreaRect(v_con_4)
            v_box = cv2.boxPoints(v_rec)
            v_box = np.int32(v_box)
            v_w_2, v_h_2 = v_rec[1]
            if v_w_2 > 0 and v_h_2 > 0:
                v_asprat = max(v_w_2, v_h_2) / min(v_w_2, v_h_2)
                if v_asprat < 2.5:
                    v_blasqu.append(
                        {
                            "type": "black_square",
                            "contour": v_con_4,
                            "box": v_box,
                            "area": v_are,
                            "aspect_ratio": v_asprat,
                            "center": v_rec[0],
                            "side_length": math.sqrt(v_are),
                        }
                    )
                    cv2.drawContours(v_debimg, [v_box], -1, (255, 0, 255), 3)
                    cv2.circle(
                        v_debimg,
                        (int(v_rec[0][0]), int(v_rec[0][1])),
                        8,
                        (255, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        v_debimg,
                        f"ORIG:{int(math.sqrt(v_are))}",
                        (int(v_rec[0][0]) - 30, int(v_rec[0][1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        2,
                    )
    return (v_blasqu, v_watdeb)


def fn_filblasqui(
    v_blasqu: List[Dict[str, Any]], v_qua_2: List[np.ndarray]
) -> List[Dict[str, Any]]:
    if not v_qua_2 or not v_blasqu:
        return []
    v_filsqu = []
    for v_squ in v_blasqu:
        v_ctr = v_squ["center"]
        for v_qua in v_qua_2:
            v_quacon = v_qua.reshape(-1, 1, 2)
            if cv2.pointPolygonTest(v_quacon, v_ctr, False) >= 0:
                v_squ["parent_quad"] = v_qua.tolist()
                v_filsqu.append(v_squ)
                break
    return v_filsqu


def fn_fms(v_filsqu: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not v_filsqu:
        return None
    v_mnsqu = min(v_filsqu, key=lambda v_x: v_x["area"])
    v_mnsqu["is_minimum"] = True
    return v_mnsqu


def fn_dms(v_img_2: np.ndarray, v_mnsqu: Optional[Dict[str, Any]]) -> None:
    if v_mnsqu is None:
        return
    v_box = v_mnsqu["box"]
    v_ctr = v_mnsqu["center"]
    cv2.drawContours(v_img_2, [v_box], -1, (255, 255, 0), 4)
    cv2.circle(v_img_2, (int(v_ctr[0]), int(v_ctr[1])), 12, (0, 255, 255), -1)
    cv2.putText(
        v_img_2,
        f"MIN BLACK:{int(v_mnsqu['side_length'])}",
        (int(v_ctr[0]) - 60, int(v_ctr[1]) - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        v_img_2,
        f"Area:{int(v_mnsqu['area'])}",
        (int(v_ctr[0]) - 40, int(v_ctr[1]) + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        2,
    )
    v_arrend = (int(v_ctr[0]), int(v_ctr[1]) - 50)
    cv2.arrowedLine(
        v_img_2,
        v_arrend,
        (int(v_ctr[0]), int(v_ctr[1]) - 15),
        (0, 255, 255),
        3,
        tipLength=0.3,
    )
    cv2.putText(
        v_img_2,
        "MINIMUM",
        (int(v_ctr[0]) - 35, int(v_ctr[1]) - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )


def fn_emssl(v_mnsqu: Optional[Dict[str, Any]]) -> int:
    if v_mnsqu is None:
        return 0
    return int(v_mnsqu["side_length"])


v_pret: float = time.time()


def fn_caploo() -> None:
    global v_frm, v_latsta, v_pret, v_vis, v_msk, v_shueve, v_capo
    while not v_shueve.is_set():
        if not v_capo:
            time.sleep(1)
            continue
        try:
            v_ret, v_img_2 = v_cap.read()
            if not v_ret:
                if v_shueve.is_set():
                    break
                time.sleep(0.01)
                continue
            with v_lk:
                v_frm = v_img_2.copy()
            v_vis, v_msk, v_sta = fn_daa(v_img_2.copy())
            v_totpxs = v_img_2.shape[0] * v_img_2.shape[1]
            v_blapxs = int((v_msk == 255).sum())
            v_now = time.time()
            v_fps = 1 / (v_now - v_pret) if v_now != v_pret else 0
            v_pret = v_now
            v_frmrat = v_sta["area_sum"] / v_totpxs * 100
            v_blarat = v_blapxs / v_totpxs * 100
            v_msi_2 = v_sta.get("minimum_black_square", {"found": False})
            v_msl = (
                int(v_msi_2.get("side_length", 0)) if v_msi_2.get("found", False) else 0
            )
            v_latsta = {
                "count": v_sta["count"],
                "total_pixels": int(v_totpxs),
                "frame_ratio": int(v_frmrat),
                "black_ratio": int(v_blarat),
                "fps": int(v_fps),
                "rects": v_sta["rects"],
                "inner_rectangles": v_sta.get("inner_rectangles", []),
                "inner_count": v_sta.get("inner_count", 0),
                "inner_total_area": v_sta.get("area_sum", 0),
                "crops_count": v_sta.get("crops_count", 0),
                "black_squares": v_sta.get("black_squares", []),
                "black_squares_count": len(v_sta.get("black_squares", [])),
                "minimum_black_square": v_sta.get(
                    "minimum_black_square", {"found": False}
                ),
                "minimum_side_length": v_msl,
            }
            if v_shueve.wait(0.03):
                break
        except Exception as v_e:
            if not v_shueve.is_set():
                print(eval('str("DEBUG")'))
                time.sleep(0.1)
            break
    print(eval('str("DEBUG")'))


v_capthr = threading.Thread(target=fn_caploo, daemon=True)
v_capthr.start()


async def fn_broloo() -> None:
    global v_shueve
    while not v_shueve.is_set():
        try:
            v_sta_2 = v_latsta.copy()
            if "rects" in v_sta_2.keys():
                for v_ite in v_sta_2["rects"]:
                    try:
                        v_ite.pop("outer_pts")
                        v_ite.pop("midpoints")
                        v_ite.pop("inner_contour")
                    except Exception:
                        pass
            v_tex = json.dumps(v_sta_2)
            v_bad = []
            for v_ws in v_cli:
                try:
                    await v_ws.send_text(v_tex)
                except WebSocketDisconnect:
                    v_bad.append(v_ws)
                except Exception as v_e:
                    if not v_shueve.is_set():
                        print(eval('str("DEBUG")'))
                    v_bad.append(v_ws)
            for v_ws in v_bad:
                v_cli.remove(v_ws)
            try:
                await asyncio.wait_for(asyncio.sleep(0.1), timeout=0.1)
            except asyncio.TimeoutError:
                pass
        except Exception as v_e:
            if not v_shueve.is_set():
                print(eval('str("DEBUG")'))
            break
    print(eval('str("DEBUG")'))


def fn_mjpgen(v_pro: str) -> Generator[bytes, None, None]:
    global v_watimg, v_msk, v_vis, v_shueve
    while not v_shueve.is_set():
        with v_lk:
            v_img_2 = v_frm.copy() if v_frm is not None else None
        if v_img_2 is None:
            if v_shueve.wait(0.01):
                break
            continue
        try:
            if v_pro == "vis":
                v_tgt = v_vis
            elif v_pro == "mask":
                v_tgt = v_msk
            elif v_pro == "watershed_img":
                if v_watimg is not None:
                    v_tgt = v_watimg
                else:
                    if v_shueve.wait(0.03):
                        break
                    continue
            else:
                if v_shueve.wait(0.03):
                    break
                continue
            if v_tgt is not None:
                v__, v_buf = cv2.imencode(".jpg", v_tgt)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + v_buf.tobytes()
                    + b"\r\n"
                )
        except Exception as v_e:
            if not v_shueve.is_set():
                print(eval('str("DEBUG")'))
            break
        if v_shueve.wait(0.03):
            break
    print(eval('str("DEBUG")'))


@v_app.get("/ocr", tags=["OCR识别"])
async def fn_detnum():
    global v_rawcro, v_capo
    v_capo = False
    v_ocrres = dict()
    if v_rawcro and v_par["enable_ocr"]:
        for v_idx, v_img in enumerate(v_rawcro):
            v_ocrres[v_idx] = fn_roof(v_img)
    v_capo = True
    return v_ocrres


@v_app.get("/api/ina226/measurements", tags=["功率监控"])
async def fn_gi2sa():
    global v_ina
    return {"data": v_ina.get_all_measurements()}


@v_app.get("/api/ocr_measurement_analysis", tags=["OCR识别"])
async def fn_goma_2():
    global v_latsta, v_rawcro, v_capo
    v_stat = time.time()
    v_capo = False
    v_meares = await fn_gpm()
    if not v_meares["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }
    v_ocrres_2 = {}
    if v_rawcro and v_par["enable_ocr"]:
        for v_idx, v_img in enumerate(v_rawcro):
            v_ocrres_2[v_idx] = fn_roof(v_img)
    v_ana = []
    for v_mea in v_meares["measurements"]:
        v_croidx = v_mea["crop_index"]
        v_croana = {
            "crop_index": v_croidx,
            "target": v_mea["target"],
            "shapes": [],
            "ocr_raw_data": v_ocrres_2.get(v_croidx, []),
        }
        for v_shp in v_mea["shapes"]:
            v_ocrdat = {"detected": False, "text": "", "confidence": 0.0, "bbox": []}
            if v_croidx in v_ocrres_2:
                v_shpbbox = v_shp.get("pixel_dimensions", {})
                v_shpctr = v_shp.get("position", {}).get("center", [0, 0])
                if v_shpctr == [0, 0]:
                    v_bbox = v_shp.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if v_bbox != [0, 0, 0, 0] and len(v_bbox) == 4:
                        v_shpctr = [
                            v_bbox[0] + v_bbox[2] / 2,
                            v_bbox[1] + v_bbox[3] / 2,
                        ]
                    else:
                        v_shpctr = [0, 0]
                if v_shpctr != [0, 0]:
                    v_swh_2 = v_shpbbox.get("width", 0) / 2
                    v_shh_2 = v_shpbbox.get("height", 0) / 2
                    for v_ocrres in v_ocrres_2[v_croidx]:
                        v_ocrbbox = v_ocrres["bbox"]
                        v_ocrctr = [
                            (v_ocrbbox[0][0] + v_ocrbbox[2][0]) / 2,
                            (v_ocrbbox[0][1] + v_ocrbbox[2][1]) / 2,
                        ]
                        v_disx = abs(v_ocrctr[0] - v_shpctr[0])
                        v_disy = abs(v_ocrctr[1] - v_shpctr[1])
                        if v_disx < v_swh_2 and v_disy < v_shh_2:
                            v_ocrdat = {
                                "detected": True,
                                "text": v_ocrres["text"],
                                "confidence": v_ocrres["conf"],
                                "bbox": v_ocrres["bbox"],
                            }
                            break
            if "position" not in v_shp:
                v_shp["position"] = {}
            v_shp["position"]["contour_points"] = v_shp.get("contour_points", [])
            v_shpana = {**v_shp, "ocr_data": v_ocrdat}
            v_croana["shapes"].append(v_shpana)
        v_ana.append(v_croana)
    v_elat = time.time() - v_stat
    v_capo = True
    return {
        "success": True,
        "analysis": v_ana,
        "total_crops": len(v_ana),
        "references": v_meares.get("a4_reference", {}),
        "elapsed_seconds": round(v_elat, 3),
    }


@v_app.get("/api/ocr_masked_analysis", tags=["OCR识别"])
async def fn_goma():
    global v_rawcro, v_capo
    v_stat = time.time()
    v_capo = False
    if not v_rawcro:
        return {"success": False, "error": "No crops data available", "ocr_results": {}}
    v_ocrres_2 = {}
    if v_par["enable_ocr"]:
        for v_idx, v_cro in enumerate(v_rawcro):
            v_mascro = fn_amtc(v_cro)
            v_rescro = cv2.resize(v_mascro, (840, 1118))
            v_ocrres_2[v_idx] = fn_roof(v_rescro)
    v_elat = time.time() - v_stat
    v_capo = True
    return {
        "success": True,
        "ocr_results": v_ocrres_2,
        "total_crops": len(v_ocrres_2),
        "elapsed_seconds": round(v_elat, 3),
    }


@v_app.get("/api/ocr_masked_measurement_analysis", tags=["OCR识别"])
async def fn_gomma():
    global v_latsta, v_rawcro, v_capo
    v_stat = time.time()
    v_capo = False
    v_meares = await fn_gpm()
    if not v_meares["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }
    v_ocrres_2 = {}
    v_scafac = {}
    if v_rawcro and v_par["enable_ocr"]:
        for v_idx, v_cro in enumerate(v_rawcro):
            v_mascro = fn_amtc(v_cro)
            v_orih, v_oriw = v_mascro.shape[:2]
            v_rescro = cv2.resize(v_mascro, (840, 1118))
            v_scax = 840 / v_oriw
            v_scay = 1118 / v_orih
            v_scafac[v_idx] = {"scale_x": v_scax, "scale_y": v_scay}
            v_ocrres_2[v_idx] = fn_roof(v_rescro)
    v_ana = []
    for v_mea in v_meares["measurements"]:
        v_croidx = v_mea["crop_index"]
        v_croana = {
            "crop_index": v_croidx,
            "target": v_mea["target"],
            "shapes": [],
            "ocr_raw_data": v_ocrres_2.get(v_croidx, []),
            "scale_factors": v_scafac.get(v_croidx, {"scale_x": 1.0, "scale_y": 1.0}),
        }
        for v_shp in v_mea["shapes"]:
            v_ocrdat = {"detected": False, "text": "", "confidence": 0.0, "bbox": []}
            if v_croidx in v_ocrres_2:
                v_shpbbox = v_shp.get("pixel_dimensions", {})
                v_shpctr = v_shp.get("position", {}).get("center", [0, 0])
                if v_shpctr == [0, 0]:
                    v_bbox = v_shp.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if v_bbox != [0, 0, 0, 0] and len(v_bbox) == 4:
                        v_shpctr = [
                            v_bbox[0] + v_bbox[2] / 2,
                            v_bbox[1] + v_bbox[3] / 2,
                        ]
                    else:
                        v_shpctr = [0, 0]
                if v_shpctr != [0, 0] and v_croidx in v_scafac:
                    v_scax = v_scafac[v_croidx]["scale_x"]
                    v_scay = v_scafac[v_croidx]["scale_y"]
                    v_scx = v_shpctr[0] * v_scax
                    v_scy = v_shpctr[1] * v_scay
                    v_swh = v_shpbbox.get("width", 0) * v_scax / 2
                    v_shh = v_shpbbox.get("height", 0) * v_scay / 2
                    for v_ocrres in v_ocrres_2[v_croidx]:
                        v_ocrbbox = v_ocrres["bbox"]
                        v_ocrctr = [
                            (v_ocrbbox[0][0] + v_ocrbbox[2][0]) / 2,
                            (v_ocrbbox[0][1] + v_ocrbbox[2][1]) / 2,
                        ]
                        v_disx = abs(v_ocrctr[0] - v_scx)
                        v_disy = abs(v_ocrctr[1] - v_scy)
                        if v_disx < v_swh and v_disy < v_shh:
                            v_oribbox = []
                            for v_pt in v_ocrbbox:
                                v_oript = [v_pt[0] / v_scax, v_pt[1] / v_scay]
                                v_oribbox.append(v_oript)
                            v_ocrdat = {
                                "detected": True,
                                "text": v_ocrres["text"],
                                "confidence": v_ocrres["conf"],
                                "bbox": v_oribbox,
                                "scaled_bbox": v_ocrres["bbox"],
                            }
                            break
            if "position" not in v_shp:
                v_shp["position"] = {}
            v_shp["position"]["contour_points"] = v_shp.get("contour_points", [])
            v_shpana = {**v_shp, "ocr_data": v_ocrdat}
            v_croana["shapes"].append(v_shpana)
        v_ana.append(v_croana)
    v_elat = time.time() - v_stat
    v_capo = True
    return {
        "success": True,
        "analysis": v_ana,
        "total_crops": len(v_ana),
        "references": v_meares.get("a4_reference", {}),
        "elapsed_seconds": round(v_elat, 3),
    }


@v_app.get("/api/ocr_scaled_measurement_analysis", tags=["OCR识别"])
async def fn_gosma():
    global v_latsta, v_rawcro, v_capo
    v_stat = time.time()
    v_capo = False
    v_meares = await fn_gpm()
    if not v_meares["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }
    v_ocrres_2 = {}
    v_scafac = {}
    if v_rawcro and v_par["enable_ocr"]:
        for v_idx, v_cro in enumerate(v_rawcro):
            v_orih, v_oriw = v_cro.shape[:2]
            v_rescro = cv2.resize(v_cro, (840, 1118))
            v_scax = 840 / v_oriw
            v_scay = 1118 / v_orih
            v_scafac[v_idx] = {"scale_x": v_scax, "scale_y": v_scay}
            v_ocrres_2[v_idx] = fn_roof(v_rescro)
    v_ana = []
    for v_mea in v_meares["measurements"]:
        v_croidx = v_mea["crop_index"]
        v_croana = {
            "crop_index": v_croidx,
            "target": v_mea["target"],
            "shapes": [],
            "ocr_raw_data": v_ocrres_2.get(v_croidx, []),
            "scale_factors": v_scafac.get(v_croidx, {"scale_x": 1.0, "scale_y": 1.0}),
        }
        for v_shp in v_mea["shapes"]:
            v_ocrdat = {"detected": False, "text": "", "confidence": 0.0, "bbox": []}
            if v_croidx in v_ocrres_2:
                v_shpbbox = v_shp.get("pixel_dimensions", {})
                v_shpctr = v_shp.get("position", {}).get("center", [0, 0])
                if v_shpctr == [0, 0]:
                    v_bbox = v_shp.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if v_bbox != [0, 0, 0, 0] and len(v_bbox) == 4:
                        v_shpctr = [
                            v_bbox[0] + v_bbox[2] / 2,
                            v_bbox[1] + v_bbox[3] / 2,
                        ]
                    else:
                        v_shpctr = [0, 0]
                if v_shpctr != [0, 0] and v_croidx in v_scafac:
                    v_scax = v_scafac[v_croidx]["scale_x"]
                    v_scay = v_scafac[v_croidx]["scale_y"]
                    v_scx = v_shpctr[0] * v_scax
                    v_scy = v_shpctr[1] * v_scay
                    v_swh = v_shpbbox.get("width", 0) * v_scax / 2
                    v_shh = v_shpbbox.get("height", 0) * v_scay / 2
                    for v_ocrres in v_ocrres_2[v_croidx]:
                        v_ocrbbox = v_ocrres["bbox"]
                        v_ocrctr = [
                            (v_ocrbbox[0][0] + v_ocrbbox[2][0]) / 2,
                            (v_ocrbbox[0][1] + v_ocrbbox[2][1]) / 2,
                        ]
                        v_disx = abs(v_ocrctr[0] - v_scx)
                        v_disy = abs(v_ocrctr[1] - v_scy)
                        if v_disx < v_swh and v_disy < v_shh:
                            v_oribbox = []
                            for v_pt in v_ocrbbox:
                                v_oript = [v_pt[0] / v_scax, v_pt[1] / v_scay]
                                v_oribbox.append(v_oript)
                            v_ocrdat = {
                                "detected": True,
                                "text": v_ocrres["text"],
                                "confidence": v_ocrres["conf"],
                                "bbox": v_oribbox,
                                "scaled_bbox": v_ocrres["bbox"],
                            }
                            break
            if "position" not in v_shp:
                v_shp["position"] = {}
            v_shp["position"]["contour_points"] = v_shp.get("contour_points", [])
            v_shpana = {**v_shp, "ocr_data": v_ocrdat}
            v_croana["shapes"].append(v_shpana)
        v_ana.append(v_croana)
    v_elat = time.time() - v_stat
    v_capo = True
    return {
        "success": True,
        "analysis": v_ana,
        "total_crops": len(v_ana),
        "references": v_meares.get("a4_reference", {}),
        "elapsed_seconds": round(v_elat, 3),
    }


@v_app.get("/api/scaled_crops/{crop_index}", tags=["图像获取"])
async def fn_gsc(v_croidx_2: int):
    global v_rawcro
    if not v_rawcro or v_croidx_2 < 0 or v_croidx_2 >= len(v_rawcro):
        raise HTTPException(status_code=404, detail="Crop not found")
    v_cro = v_rawcro[v_croidx_2]
    v_rescro = cv2.resize(v_cro, (840, 1118))
    v__, v_buf_2 = cv2.imencode(".jpg", v_rescro)
    return StreamingResponse(
        io.BytesIO(v_buf_2.tobytes()),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"inline; filename=scaled_crop_{v_croidx_2}.jpg"
        },
    )


@v_app.get("/api/masked_crops/{crop_index}", tags=["图像获取"])
async def fn_gmc(v_croidx_2: int):
    global v_rawcro
    if not v_rawcro or v_croidx_2 < 0 or v_croidx_2 >= len(v_rawcro):
        raise HTTPException(status_code=404, detail="Crop not found")
    v_cro = v_rawcro[v_croidx_2]
    v_mascro = fn_amtc(v_cro)
    v_rescro = cv2.resize(v_mascro, (840, 1118))
    v__, v_buf_2 = cv2.imencode(".jpg", v_rescro)
    return StreamingResponse(
        io.BytesIO(v_buf_2.tobytes()),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"inline; filename=masked_crop_{v_croidx_2}.jpg"
        },
    )


def fn_amtc(v_cro: np.ndarray) -> np.ndarray:
    v_pro_2, v__ = fn_preimg(v_cro)
    return v_pro_2


@v_app.get("/video/processed", tags=["视频流"])
def fn_vidpro() -> StreamingResponse:
    return StreamingResponse(
        fn_mjpgen("vis"), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@v_app.get("/video/mask", tags=["视频流"])
def fn_vidmsk() -> StreamingResponse:
    return StreamingResponse(
        fn_mjpgen("mask"), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@v_app.get("/video/watershed_img", tags=["视频流"])
def fn_vwi() -> StreamingResponse:
    return StreamingResponse(
        fn_mjpgen("watershed_img"),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@v_app.post("/control/hsv", tags=["参数控制"])
async def fn_sethsv(v_req: Request) -> Dict[str, Any]:
    v_d = await v_req.json()
    for v_k in [
        "h1_min",
        "h1_max",
        "s1_min",
        "s1_max",
        "v1_min",
        "v1_max",
        "h2_min",
        "h2_max",
        "s2_min",
        "s2_max",
        "v2_min",
        "v2_max",
    ]:
        if v_k not in v_d:
            raise HTTPException(400)
        v_par[v_k] = int(v_d[v_k])
    v_par["use_range2"] = bool(v_d.get("use_range2", 0))
    v_par["min_area"] = int(v_d.get("min_area", 200))
    return {"params": v_par}


@v_app.post("/control/canny", tags=["参数控制"])
async def fn_setcan(v_req: Request) -> Dict[str, Any]:
    v_d = await v_req.json()
    for v_k in ["canny_min", "canny_max"]:
        if v_k not in v_d:
            raise HTTPException(400)
        v_par[v_k] = int(v_d[v_k])
    return {"params": v_par}


@v_app.post("/control/show_rectangles")
async def fn_tsr(v_req: Request) -> Dict[str, bool]:
    global v_sar
    v_d = await v_req.json()
    v_sar = bool(v_d.get("show", False))
    return {"show_all_rectangles": v_sar}


@v_app.get("/control/show_rectangles")
async def fn_gsr() -> Dict[str, bool]:
    return {"show_all_rectangles": v_sar}


@v_app.get("/api/minimum_square", tags=["形状检测"])
async def fn_gms() -> Dict[str, Any]:
    return v_latsta.get("minimum_black_square", {"found": False})


@v_app.get("/api/minimum_square/side_length", tags=["形状检测"])
async def fn_gmssl() -> Dict[str, int]:
    v_msi_2 = v_latsta.get("minimum_black_square", {"found": False})
    if v_msi_2.get("found", False):
        return {"side_length": int(v_msi_2.get("side_length", 0))}
    else:
        return {"side_length": 0}


@v_app.get("/api/inner_rectangles")
async def fn_gir() -> Dict[str, Any]:
    return {
        "inner_count": v_latsta.get("inner_count", 0),
        "inner_rectangles": v_latsta.get("inner_rectangles", []),
        "inner_total_area": v_latsta.get("inner_total_area", 0),
    }


@v_app.get("/api/inner_rectangles/count")
async def fn_girc() -> Dict[str, int]:
    return {"count": v_latsta.get("inner_count", 0)}


@v_app.get("/api/inner_rectangles/crops")
async def fn_gic() -> Dict[str, Any]:
    with v_lk:
        v_cro_2 = v_curcro
        v_crodat = []
        for v_i, v_cro in enumerate(v_cro_2):
            v__, v_buf_2 = cv2.imencode(".jpg", v_cro)
            v_ib6 = base64.b64encode(v_buf_2).decode("utf-8")
            v_crodat.append({"index": v_i, "shape": v_cro.shape, "image_base64": v_ib6})
    return {
        "crops_count": len(v_crodat),
        "crops": v_crodat,
        "timestamp": datetime.now().isoformat(),
    }


@v_app.websocket("/ws")
async def fn_wsep(v_ws: WebSocket) -> None:
    await v_ws.accept()
    v_cli.append(v_ws)
    try:
        while True:
            await v_ws.receive_text()
    except WebSocketDisconnect:
        v_cli.remove(v_ws)


@v_app.on_event("startup")
async def fn_sta() -> None:
    print(eval('str("DEBUG")'))
    asyncio.create_task(fn_broloo())


@v_app.on_event("shutdown")
async def fn_shu() -> None:
    print(eval('str("DEBUG")'))
    global v_shueve, v_capthr, v_cap
    v_shueve.set()
    if v_capthr and v_capthr.is_alive():
        print(eval('str("DEBUG")'))
        v_capthr.join(timeout=2.0)
        if v_capthr.is_alive():
            print(eval('str("DEBUG")'))
    if v_cap and v_cap.isOpened():
        print(eval('str("DEBUG")'))
        v_cap.release()
    for v_ws in v_cli.copy():
        try:
            await v_ws.close()
        except Exception:
            pass
    v_cli.clear()
    print(eval('str("DEBUG")'))


@v_app.get("/debug/area", response_class=HTMLResponse)
def fn_arecon() -> HTMLResponse:
    with open("area_filter_control.html", "r", encoding="utf-8") as v_f:
        return HTMLResponse(v_f.read())


@v_app.get("/debug/area2", response_class=HTMLResponse)
def fn_ac2() -> HTMLResponse:
    with open("a4_measurement_control.html", "r", encoding="utf-8") as v_f:
        return HTMLResponse(v_f.read())


@v_app.get("/config")
async def fn_getcon() -> Dict[str, Any]:
    return {
        "detection_params": v_par,
        "algorithm_params": v_detpar,
        "area_filter_params": v_afp,
        "perspective_params": v_perpar,
        "black_detection_params": v_bdp,
        "camera_params": v_camcon,
    }


@v_app.post("/config/detection")
async def fn_udc(v_dat: Dict[str, Any]) -> Dict[str, Any]:
    global v_par
    try:
        for v_key, v_val in v_dat.items():
            if v_key in v_par:
                v_par[v_key] = v_val
        fn_savcon()
        return {"success": True, "message": "检测参数已更新", "detection_params": v_par}
    except Exception as v_e:
        return {"success": False, "message": f"更新失败: {str(v_e)}"}


@v_app.post("/config/black_detection")
async def fn_ubdc(v_dat: Dict[str, Any]) -> Dict[str, Any]:
    global v_bdp
    try:
        for v_key, v_val in v_dat.items():
            if v_key in v_bdp:
                v_bdp[v_key] = v_val
        fn_savcon()
        return {
            "success": True,
            "message": "黑色检测参数已更新",
            "black_detection_params": v_bdp,
        }
    except Exception as v_e:
        return {"success": False, "message": f"更新失败: {str(v_e)}"}


@v_app.post("/config/detection_params")
async def fn_udpc(v_dat: Dict[str, Any]) -> Dict[str, Any]:
    global v_detpar
    try:
        for v_key, v_val in v_dat.items():
            if v_key in v_detpar:
                v_detpar[v_key] = v_val
        v_con_2 = fn_loacon()
        if "min_vertices" in v_dat:
            v_con_2["detection"]["min_vertices"] = v_dat["min_vertices"]
        if "max_vertices" in v_dat:
            v_con_2["detection"]["max_vertices"] = v_dat["max_vertices"]
        fn_savcon(v_con_2)
        return {
            "success": True,
            "message": "检测算法参数已更新",
            "detection_params": v_detpar,
        }
    except Exception as v_e:
        return {"success": False, "message": f"更新失败: {str(v_e)}"}


@v_app.post("/config/area_filter")
async def fn_uafc(v_dat: Dict[str, Any]) -> Dict[str, Any]:
    global v_afp
    try:
        for v_key, v_val in v_dat.items():
            if v_key in v_afp:
                v_afp[v_key] = v_val
        fn_savcon()
        return {
            "success": True,
            "message": "面积过滤参数已更新",
            "area_filter_params": v_afp,
        }
    except Exception as v_e:
        return {"success": False, "message": f"更新失败: {str(v_e)}"}


@v_app.post("/config/perspective")
async def fn_upc(v_dat: Dict[str, Any]) -> Dict[str, Any]:
    global v_perpar
    try:
        for v_key, v_val in v_dat.items():
            if v_key in v_perpar:
                v_perpar[v_key] = v_val
        fn_savcon()
        return {
            "success": True,
            "message": "梯形校正参数已更新",
            "perspective_params": v_perpar,
        }
    except Exception as v_e:
        return {"success": False, "message": f"更新失败: {str(v_e)}"}


@v_app.post("/config/camera")
async def fn_ucc(v_dat: Dict[str, Any]) -> Dict[str, Any]:
    global v_camcon
    try:
        for v_key, v_val in v_dat.items():
            if v_key in v_camcon:
                v_camcon[v_key] = v_val
        fn_savcon()
        return {
            "success": True,
            "message": "摄像头参数已更新",
            "camera_params": v_camcon,
        }
    except Exception as v_e:
        return {"success": False, "message": f"更新失败: {str(v_e)}"}


@v_app.post("/config/custom_string")
async def fn_scs(v_req_2: Request) -> Dict[str, Any]:
    try:
        v_dat = await v_req_2.json()
        if "custom_config" not in v_par:
            v_par["custom_config"] = {}
        if "key" in v_dat and "value" in v_dat:
            v_key = str(v_dat["key"])
            v_par["custom_config"][v_key] = v_dat["value"]
        elif "configs" in v_dat and isinstance(v_dat["configs"], list):
            for v_ite in v_dat["configs"]:
                if isinstance(v_ite, dict) and "key" in v_ite and ("value" in v_ite):
                    v_key = str(v_ite["key"])
                    v_par["custom_config"][v_key] = v_ite["value"]
                else:
                    raise ValueError("Invalid config format in batch update")
        else:
            raise ValueError("Invalid request format")
        fn_savcon()
        return {
            "success": True,
            "message": "自定义配置已保存",
            "custom_config": v_par["custom_config"],
        }
    except Exception as v_e:
        return {"success": False, "message": f"保存失败: {str(v_e)}"}


@v_app.get("/config/custom_string")
async def fn_gcs() -> Dict[str, Any]:
    try:
        if "custom_config" not in v_par:
            v_par["custom_config"] = {}
        return {"success": True, "custom_config": v_par["custom_config"]}
    except Exception as v_e:
        return {"success": False, "message": f"获取失败: {str(v_e)}"}


@v_app.get("/crops")
async def fn_getcro() -> Dict[str, Any]:
    global v_curcro
    if not v_curcro:
        return {"crops_count": 0, "crops": []}
    v_croinf = []
    for v_i, v_cro in enumerate(v_curcro):
        v_h_2, v_w_2 = v_cro.shape[:2]
        v_croinf.append(
            {
                "index": v_i,
                "width": v_w_2,
                "height": v_h_2,
                "channels": v_cro.shape[2] if len(v_cro.shape) > 2 else 1,
            }
        )
    return {"crops_count": len(v_curcro), "crops": v_croinf}


@v_app.get("/crop/{crop_index}")
async def fn_gci(v_croidx_2: int) -> Response:
    global v_curcro
    if not v_curcro or v_croidx_2 < 0 or v_croidx_2 >= len(v_curcro):
        raise HTTPException(status_code=404, detail="裁剪图像不存在")
    v_cro = v_curcro[v_croidx_2]
    v__, v_buf_2 = cv2.imencode(".jpg", v_cro)
    return Response(
        content=v_buf_2.tobytes(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=crop_{v_croidx_2}.jpg"},
    )


@v_app.get("/rawcrop/{crop_index}")
async def fn_gri(v_croidx_2: int) -> Response:
    global v_rawcro
    if not v_rawcro or v_croidx_2 < 0 or v_croidx_2 >= len(v_rawcro):
        raise HTTPException(status_code=404, detail="裁剪图像不存在")
    v_cro = v_rawcro[v_croidx_2]
    v__, v_buf_2 = cv2.imencode(".jpg", v_cro)
    return Response(
        content=v_buf_2.tobytes(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=crop_{v_croidx_2}.jpg"},
    )


@v_app.get("/crop/min/{crop_index}")
async def fn_gmsi(v_croidx_2: int) -> Response:
    global v_msi
    if not v_msi or v_croidx_2 < 0 or v_croidx_2 >= len(v_msi):
        raise HTTPException(status_code=404, detail="最小正方形检测图像不存在")
    v_mnimg = v_msi[v_croidx_2]
    v__, v_buf_2 = cv2.imencode(".jpg", v_mnimg)
    return Response(
        content=v_buf_2.tobytes(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"inline; filename=min_square_{v_croidx_2}.jpg"
        },
    )


@v_app.get("/api/physical_measurements")
async def fn_gpm():
    global v_latsta
    if not v_latsta or "rects" not in v_latsta:
        return {
            "success": False,
            "error": "No measurement data available",
            "measurements": [],
        }
    v_mea_2 = []
    for v_croidx, v_recdat in enumerate(v_latsta["rects"]):
        v_cromea_2 = {
            "crop_index": v_croidx,
            "target": {
                "id": v_recdat["id"],
                "bbox": v_recdat.get(
                    "bbox",
                    [
                        v_recdat["position"][0],
                        v_recdat["position"][1],
                        v_recdat["outer_width"],
                        v_recdat["outer_height"],
                    ],
                ),
                "area": v_recdat["area"],
                "aspect_ratio": v_recdat["aspect_ratio"],
                "crop_width": v_recdat["crop_width"],
                "crop_height": v_recdat["crop_height"],
                "position": v_recdat["position"],
                "horizontal_avg": v_recdat["horizontal_avg"],
                "vertical_avg": v_recdat["vertical_avg"],
                "new_long_px": v_recdat["new_long_px"],
            },
            "shapes": [],
        }
        if "all_shapes" in v_recdat and v_recdat["all_shapes"]:
            for v_shpidx, v_shpdat in enumerate(v_recdat["all_shapes"]):
                if "physical_info" in v_shpdat:
                    v_phyinf = v_shpdat["physical_info"]
                    v_shpmea = {
                        "shape_index": v_shpidx,
                        "shape_type": v_shpdat["shape_type"],
                        "pixel_dimensions": {
                            "width": v_shpdat["width"],
                            "height": v_shpdat["height"],
                            "area": v_shpdat["area"],
                            "side_lengths": v_shpdat.get("side_lengths", []),
                            "mean_side_length": v_shpdat.get("mean_side_length", 0),
                            "perimeter": v_shpdat.get("perimeter", 0),
                        },
                        "physical_dimensions": {
                            "width_mm": v_phyinf["physical_width_mm"],
                            "height_mm": v_phyinf["physical_height_mm"],
                            "area_mm2": v_phyinf["physical_area_mm2"],
                            "diameter_mm": v_phyinf["physical_diameter_mm"],
                            "side_lengths_mm": v_phyinf["physical_side_lengths_mm"],
                            "perimeter_mm": v_phyinf["physical_perimeter_mm"],
                            "measurement_type": v_phyinf["measurement_type"],
                            "mm_per_pixel": v_phyinf["mm_per_pixel"],
                        },
                        "position": v_shpdat.get(
                            "position",
                            {
                                "center": [0, 0],
                                "bbox": [0, 0, 0, 0],
                                "contour_points": [],
                            },
                        ),
                    }
                    v_cromea_2["shapes"].append(v_shpmea)
        v_mea_2.append(v_cromea_2)
    return {
        "success": True,
        "measurements": v_mea_2,
        "total_crops": len(v_mea_2),
        "a4_reference": {
            "physical_width_mm": 170,
            "physical_height_mm": 257,
            "note": "A4 paper minus 20mm border on each side",
        },
    }


@v_app.get("/api/physical_measurements/{crop_index}")
async def fn_gcpm(v_croidx_2: int):
    global v_latsta
    if (
        not v_latsta
        or "rects" not in v_latsta
        or v_croidx_2 < 0
        or (v_croidx_2 >= len(v_latsta["rects"]))
    ):
        raise HTTPException(status_code=404, detail="Crop index not found")
    v_recdat = v_latsta["rects"][v_croidx_2]
    v_cromea_2 = {"crop_index": v_croidx_2, "shapes": []}
    if "all_shapes" in v_recdat and v_recdat["all_shapes"]:
        for v_shpidx, v_shpdat in enumerate(v_recdat["all_shapes"]):
            if "physical_info" in v_shpdat:
                v_phyinf = v_shpdat["physical_info"]
                v_shpmea = {
                    "shape_index": v_shpidx,
                    "shape_type": v_shpdat["shape_type"],
                    "pixel_dimensions": {
                        "width": v_shpdat["width"],
                        "height": v_shpdat["height"],
                        "area": v_shpdat["area"],
                        "side_lengths": v_shpdat.get("side_lengths", []),
                        "mean_side_length": v_shpdat.get("mean_side_length", 0),
                        "perimeter": v_shpdat.get("perimeter", 0),
                    },
                    "physical_dimensions": {
                        "width_mm": v_phyinf["physical_width_mm"],
                        "height_mm": v_phyinf["physical_height_mm"],
                        "area_mm2": v_phyinf["physical_area_mm2"],
                        "diameter_mm": v_phyinf["physical_diameter_mm"],
                        "side_lengths_mm": v_phyinf["physical_side_lengths_mm"],
                        "perimeter_mm": v_phyinf["physical_perimeter_mm"],
                        "measurement_type": v_phyinf["measurement_type"],
                        "mm_per_pixel": v_phyinf["mm_per_pixel"],
                    },
                }
                v_cromea_2["shapes"].append(v_shpmea)
    return {
        "success": True,
        "crop_measurements": v_cromea_2,
        "a4_reference": {
            "physical_width_mm": 170,
            "physical_height_mm": 257,
            "note": "A4 paper minus 20mm border on each side",
        },
    }


@v_app.get("/api/minimum_square_measurements")
async def fn_gmsm():
    global v_latsta
    if not v_latsta or "rects" not in v_latsta:
        return {
            "success": False,
            "error": "No measurement data available",
            "measurements": [],
        }
    v_mea_2 = []
    for v_croidx, v_recdat in enumerate(v_latsta["rects"]):
        v_cromea = {
            "crop_index": v_croidx,
            "target": {
                "id": v_recdat["id"],
                "bbox": [
                    v_recdat["position"][0],
                    v_recdat["position"][1],
                    v_recdat["outer_width"],
                    v_recdat["outer_height"],
                ],
                "area": v_recdat["area"],
                "aspect_ratio": v_recdat["aspect_ratio"],
                "crop_width": v_recdat["crop_width"],
                "crop_height": v_recdat["crop_height"],
                "position": v_recdat["position"],
                "horizontal_avg": v_recdat["horizontal_avg"],
                "vertical_avg": v_recdat["vertical_avg"],
                "new_long_px": v_recdat["new_long_px"],
            },
            "edges": [],
        }
        v_msi_2 = v_latsta.get("minimum_black_square", {"found": False})
        if v_msi_2.get("found", False):
            v_elp = v_msi_2["edge_length_px"]
            v_elm = v_msi_2["edge_length_mm"]
            v_a4wm = 170
            v_crow = v_recdat["crop_width"]
            if v_crow > 0:
                v_mpp = v_a4wm / v_crow
                v_squmea = {
                    "shape_index": 0,
                    "found": True,
                    "center": v_msi_2["center"],
                    "edge_length_px": v_elp,
                    "edge_length_mm": v_elm,
                    "type": v_msi_2.get("type", "minimum_edge_detected"),
                    "start_point": v_msi_2.get("start_point", [0, 0]),
                    "end_point": v_msi_2.get("end_point", [0, 0]),
                    "pixel_dimensions": {
                        "edge_length": v_elp,
                        "note": "This is the shortest detected edge, not a complete square",
                    },
                    "physical_dimensions": {
                        "edge_length_mm": v_elm,
                        "mm_per_pixel": v_mpp,
                        "note": "Physical measurement of the shortest detected edge",
                    },
                }
                v_cromea["edges"].append(v_squmea)
        else:
            v_cromea["edges"].append({"shape_index": 0, "found": False})
        v_mea_2.append(v_cromea)
    return {
        "success": True,
        "measurements": v_mea_2,
        "total_crops": len(v_mea_2),
        "a4_reference": {
            "physical_width_mm": 170,
            "physical_height_mm": 257,
            "note": "A4 paper minus 20mm border on each side",
        },
    }


@v_app.get("/api/ocr_auto_segment_easyocr_analysis", tags=["OCR识别"])
async def fn_goasea():
    global v_latsta, v_rawcro, v_capo
    v_capo = False
    v_stat = time.time()
    v_meares = await fn_gpm()
    if not v_meares["success"]:
        v_capo = True
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }
    v_asr = {}
    v_scafac = {}
    print(eval('str("DEBUG")'))
    if v_rawcro and v_par["enable_ocr"]:
        print(eval('str("DEBUG")'))
        for v_idx, v_cro in enumerate(v_rawcro):
            print(eval('str("DEBUG")'))
            try:
                v_mascro = fn_amtc(v_cro)
                v_orih, v_oriw = v_mascro.shape[:2]
                v_rescro = cv2.resize(v_mascro, (840, 1118))
                v_scax = 840 / v_oriw
                v_scay = 1118 / v_orih
                v_scafac[v_idx] = {"scale_x": v_scax, "scale_y": v_scay}
                print(eval('str("DEBUG")'))
                print(eval('str("DEBUG")'))
                print(eval('str("DEBUG")'))
                print(eval('str("DEBUG")'))
                v_asp_2 = cls_laseo()
                v_easres = v_asp_2.fn_proimg(
                    v_rescro, 0.5, True
                )
                v_conres_2 = []
                for v_res_2 in v_easres:
                    v_conres = {
                        "text": v_res_2["text"],
                        "conf": v_res_2["confidence"],
                        "center": [v_res_2["center_x"], v_res_2["center_y"]],
                        "bbox": v_res_2["bbox_in_original"],
                        "rectangle_id": v_res_2.get("rectangle_id", 0),
                        "rectangle_area": v_res_2.get("rectangle_area", 0),
                    }
                    v_conres_2.append(v_conres)
                v_asr[v_idx] = v_conres_2
                print(eval('str("DEBUG")'))
            except Exception as v_e:
                print(eval('str("DEBUG")'))
                v_asr[v_idx] = []
    else:
        print(eval('str("DEBUG")'))
    v_ana = []
    for v_mea in v_meares["measurements"]:
        v_croidx = v_mea["crop_index"]
        v_croana = {
            "crop_index": v_croidx,
            "target": v_mea["target"],
            "shapes": [],
            "ocr_raw_data": v_asr.get(v_croidx, []),
            "scale_factors": v_scafac.get(v_croidx, {"scale_x": 1.0, "scale_y": 1.0}),
        }
        for v_shp in v_mea["shapes"]:
            v_ocrdat = {
                "detected": False,
                "text": "",
                "conf": 0.0,
                "bbox": [],
                "center": [],
            }
            if v_croidx in v_asr:
                v_shpbbox = v_shp.get("pixel_dimensions", {})
                v_shpctr = v_shp.get("position", {}).get("center", [0, 0])
                if v_shpctr == [0, 0]:
                    v_bbox = v_shp.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if v_bbox != [0, 0, 0, 0] and len(v_bbox) == 4:
                        v_shpctr = [
                            v_bbox[0] + v_bbox[2] / 2,
                            v_bbox[1] + v_bbox[3] / 2,
                        ]
                    else:
                        v_shpctr = [0, 0]
                if v_shpctr != [0, 0] and v_croidx in v_scafac:
                    v_scax = v_scafac[v_croidx]["scale_x"]
                    v_scay = v_scafac[v_croidx]["scale_y"]
                    v_scx = v_shpctr[0] * v_scax
                    v_scy = v_shpctr[1] * v_scay
                    v_swh = v_shpbbox.get("width", 0) * v_scax / 2
                    v_shh = v_shpbbox.get("height", 0) * v_scay / 2
                    v_besmtc = None
                    v_mndis = float("inf")
                    for v_res_2 in v_asr[v_croidx]:
                        v_resctr = v_res_2["center"]
                        v_disx = abs(v_resctr[0] - v_scx)
                        v_disy = abs(v_resctr[1] - v_scy)
                        v_totdis = (v_disx**2 + v_disy**2) ** 0.5
                        if v_disx < v_swh and v_disy < v_shh and (v_totdis < v_mndis):
                            v_mndis = v_totdis
                            v_besmtc = v_res_2
                    if v_besmtc:
                        v_oribbox = []
                        for v_pt in v_besmtc["bbox"]:
                            v_oript = [v_pt[0] / v_scax, v_pt[1] / v_scay]
                            v_oribbox.append(v_oript)
                        v_orictr = [
                            v_besmtc["center"][0] / v_scax,
                            v_besmtc["center"][1] / v_scay,
                        ]
                        v_ocrdat = {
                            "detected": True,
                            "text": v_besmtc["text"],
                            "conf": v_besmtc["conf"],
                            "bbox": v_oribbox,
                            "center": v_orictr,
                            "scaled_bbox": v_besmtc["bbox"],
                            "scaled_center": v_besmtc["center"],
                            "rectangle_id": v_besmtc.get("rectangle_id", 0),
                            "rectangle_area": v_besmtc.get("rectangle_area", 0),
                            "match_distance": v_mndis,
                        }
            if "position" not in v_shp:
                v_shp["position"] = {}
            v_shp["position"]["contour_points"] = v_shp.get("contour_points", [])
            v_shpana = {**v_shp, "ocr_data": v_ocrdat}
            v_croana["shapes"].append(v_shpana)
        v_ana.append(v_croana)
    v_elat = time.time() - v_stat
    v_capo = True
    return {
        "success": True,
        "analysis": v_ana,
        "total_crops": len(v_ana),
        "references": v_meares.get("a4_reference", {}),
        "elapsed_seconds": round(v_elat, 3),
    }


@v_app.get("/api/ocr_auto_segment_analysis", tags=["OCR识别"])
async def fn_goasa():
    global v_latsta, v_rawcro, v_capo
    v_stat = time.time()
    v_capo = False
    v_meares = await fn_gpm()
    if not v_meares["success"]:
        return {
            "success": False,
            "error": "Failed to get physical measurements",
            "analysis": [],
        }
    v_asr = {}
    v_scafac = {}
    print(eval('str("DEBUG")'))
    if v_rawcro and v_par["enable_ocr"]:
        print(eval('str("DEBUG")'))
        for v_idx, v_cro in enumerate(v_rawcro):
            print(eval('str("DEBUG")'))
            try:
                v_mascro = fn_amtc(v_cro)
                v_orih, v_oriw = v_mascro.shape[:2]
                v_rescro = cv2.resize(v_mascro, (840, 1118))
                v_scax = 840 / v_oriw
                v_scay = 1118 / v_orih
                v_scafac[v_idx] = {"scale_x": v_scax, "scale_y": v_scay}
                v__, v_buf_2 = cv2.imencode(".jpg", v_rescro)
                v_ib6_2 = base64.b64encode(v_buf_2).decode("utf-8")
                print(eval('str("DEBUG")'))
                print(eval('str("DEBUG")'))
                v_loo = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as v_exe:
                    try:
                        v_conres_2 = await v_loo.run_in_executor(
                            v_exe, fn_piwas, v_ib6_2, v_idx
                        )
                        v_asr[v_idx] = v_conres_2
                    except Exception as v_e:
                        print(eval('str("DEBUG")'))
                        import traceback

                        traceback.print_exc()
                        v_asr[v_idx] = []
            except Exception as v_e:
                print(eval('str("DEBUG")'))
                v_asr[v_idx] = []
    else:
        print(eval('str("DEBUG")'))
    v_ana = []
    for v_mea in v_meares["measurements"]:
        v_croidx = v_mea["crop_index"]
        v_croana = {
            "crop_index": v_croidx,
            "target": v_mea["target"],
            "shapes": [],
            "ocr_raw_data": v_asr.get(v_croidx, []),
            "scale_factors": v_scafac.get(v_croidx, {"scale_x": 1.0, "scale_y": 1.0}),
        }
        for v_shp in v_mea["shapes"]:
            v_ocrdat = {
                "detected": False,
                "text": "",
                "conf": 0.0,
                "bbox": [],
                "center": [],
            }
            if v_croidx in v_asr:
                v_shpbbox = v_shp.get("pixel_dimensions", {})
                v_shpctr = v_shp.get("position", {}).get("center", [0, 0])
                if v_shpctr == [0, 0]:
                    v_bbox = v_shp.get("position", {}).get("bbox", [0, 0, 0, 0])
                    if v_bbox != [0, 0, 0, 0] and len(v_bbox) == 4:
                        v_shpctr = [
                            v_bbox[0] + v_bbox[2] / 2,
                            v_bbox[1] + v_bbox[3] / 2,
                        ]
                    else:
                        v_shpctr = [0, 0]
                if v_shpctr != [0, 0] and v_croidx in v_scafac:
                    v_scax = v_scafac[v_croidx]["scale_x"]
                    v_scay = v_scafac[v_croidx]["scale_y"]
                    v_scx = v_shpctr[0] * v_scax
                    v_scy = v_shpctr[1] * v_scay
                    v_swh = v_shpbbox.get("width", 0) * v_scax / 2
                    v_shh = v_shpbbox.get("height", 0) * v_scay / 2
                    v_besmtc = None
                    v_mndis = float("inf")
                    for v_res_2 in v_asr[v_croidx]:
                        v_resctr = v_res_2["center"]
                        v_disx = abs(v_resctr[0] - v_scx)
                        v_disy = abs(v_resctr[1] - v_scy)
                        v_totdis = (v_disx**2 + v_disy**2) ** 0.5
                        if v_disx < v_swh and v_disy < v_shh and (v_totdis < v_mndis):
                            v_mndis = v_totdis
                            v_besmtc = v_res_2
                    if v_besmtc:
                        v_oribbox = []
                        for v_pt in v_besmtc["bbox"]:
                            v_oript = [v_pt[0] / v_scax, v_pt[1] / v_scay]
                            v_oribbox.append(v_oript)
                        v_orictr = [
                            v_besmtc["center"][0] / v_scax,
                            v_besmtc["center"][1] / v_scay,
                        ]
                        v_ocrdat = {
                            "detected": True,
                            "text": v_besmtc["text"],
                            "conf": v_besmtc["conf"],
                            "bbox": v_oribbox,
                            "center": v_orictr,
                            "scaled_bbox": v_besmtc["bbox"],
                            "scaled_center": v_besmtc["center"],
                            "rectangle_id": v_besmtc.get("rectangle_id", 0),
                            "rectangle_area": v_besmtc.get("rectangle_area", 0),
                            "match_distance": v_mndis,
                        }
            if "position" not in v_shp:
                v_shp["position"] = {}
            v_shp["position"]["contour_points"] = v_shp.get("contour_points", [])
            v_shpana = {**v_shp, "ocr_data": v_ocrdat}
            v_croana["shapes"].append(v_shpana)
        v_ana.append(v_croana)
    v_elat = time.time() - v_stat
    v_capo = True
    return {
        "success": True,
        "analysis": v_ana,
        "total_crops": len(v_ana),
        "references": v_meares.get("a4_reference", {}),
        "elapsed_seconds": round(v_elat, 3),
    }


v_stadir = os.path.join(os.path.dirname(__file__), "spa")
print(eval('str("DEBUG")'))
v_app.mount("/", StaticFiles(directory=v_stadir, html=True), name="ui")
