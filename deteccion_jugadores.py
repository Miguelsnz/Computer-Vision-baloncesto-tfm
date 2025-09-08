# deteccion_jugadores.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                        # Fuerza CPU
os.environ["TORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Memoria Torch

import torch
import cv2
import numpy as np
from operator import itemgetter

# Detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from graficos import plt_plot  # compat

# Colores HSV usados para diferenciar equipos y árbitro
COLORS = {
    'green': ([56, 50, 50], [100, 255, 255], [72, 200, 153]),  # Equipo verde
    'referee': ([0, 0, 0], [255, 35, 65], [120, 0, 0]),        # Árbitro (oscuro)
    'white': ([0, 0, 190], [255, 26, 255], [255, 0, 255])      # Equipo blanco
}

# Parámetros fijos
IOU_TH = 0.2   # Umbral para IoU al asociar jugadores detectados
PAD = 15       # Margen adicional en las cajas delimitadoras


def hsv2bgr(color_hsv):
    """Convierte HSV a BGR (OpenCV)."""
    color_bgr = np.array(cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)).ravel()
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

# Alias en español de la función libre
hsv_a_bgr = hsv2bgr


class FeetDetector:
    """
    Detecta jugadores (Mask R-CNN), estima equipo por color y proyecta posiciones al mapa 2D.
    """

    def __init__(self, players):
        cfg_seg = get_cfg()
        cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg_seg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg_seg.MODEL.DEVICE = "cpu"

        print("[DEBUG] cfg_seg.MODEL.DEVICE =", cfg_seg.MODEL.DEVICE)
        print("[DEBUG] torch.cuda.is_available() =", torch.cuda.is_available())

        self.model = build_model(cfg_seg)
        DetectionCheckpointer(self.model).load(cfg_seg.MODEL.WEIGHTS)
        self.model.eval()

        def _predict_cpu(img_bgr):
            with torch.no_grad():
                h, w = img_bgr.shape[:2]
                tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float()
                outputs = self.model([{"image": tensor, "height": h, "width": w}])[0]
                return outputs
        self.predictor_seg = _predict_cpu

        self.bbs = []
        self.players = players
        self.cfg = cfg_seg

    @staticmethod
    def count_non_black(image):
        """Cuenta el número de píxeles no negros en una imagen."""
        return np.count_nonzero(image > 0.0001)

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        """Calcula IoU entre dos cajas (x1,y1,x2,y2)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        return interArea / float(boxAArea + boxBArea - interArea)

    # ===== Alias estáticos en español =====
    contar_no_negros = count_non_black
    interseccion_sobre_union = bb_intersection_over_union

    def get_players_pos(self, M, M1, frame, timestamp, map_2d):
        """
        Detecta jugadores, asigna color/equipo y proyecta posiciones a 2D.
        """
        warped_kpts = []

        outputs_seg = self.predictor_seg(frame)
        instances = outputs_seg["instances"]
        indices = instances.pred_classes.cpu().numpy()
        predicted_masks = instances.pred_masks.cpu().numpy()

        ppl_pairs = []
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], np.uint8)

        for i, entry in enumerate(indices):
            if entry == 0:  # persona
                mask = cv2.erode(np.array(predicted_masks[i], dtype=np.uint8),
                                 kernel, iterations=4).astype(bool)
                coords = np.column_stack(np.where(mask))
                if coords.size == 0:
                    continue
                ppl_pairs.append((coords, mask))

        if not ppl_pairs:
            return frame, map_2d, map_2d.copy()

        for keypoint, p in ppl_pairs:
            top = int(np.min(keypoint[:, 0]))
            bottom = int(np.max(keypoint[:, 0]))
            left = int(np.min(keypoint[:, 1]))
            right = int(np.max(keypoint[:, 1]))
            bbox_person = (top - PAD, left - PAD, bottom + PAD, right + PAD)

            h = bottom - top
            if h <= 0 or right <= left:
                continue

            tmp_tensor = p.reshape((p.shape[0], p.shape[1], 1))
            crop_img = np.where(tmp_tensor, frame, 0)

            y1 = max(0, top)
            y2 = max(0, min(bottom - int(0.3 * h), frame.shape[0]))
            x1 = max(0, left)
            x2 = max(0, min(right, frame.shape[1]))
            if y2 <= y1 or x2 <= x1:
                continue

            crop_img = crop_img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            crop_img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            best_mask = [0, '']  # (num_non_black, color_key)
            for color_key in COLORS.keys():
                mask = cv2.inRange(crop_img_hsv,
                                   np.array(COLORS[color_key][0]),
                                   np.array(COLORS[color_key][1]))
                output = cv2.bitwise_and(crop_img_hsv, crop_img_hsv, mask=mask)
                non_blacks = FeetDetector.count_non_black(output)
                if non_blacks > best_mask[0]:
                    best_mask[0] = non_blacks
                    best_mask[1] = color_key

            head_idx = int(np.argmin(keypoint[:, 0]))
            foot_idx = int(np.argmax(keypoint[:, 0]))
            head_y, head_x = int(keypoint[head_idx, 0]), int(keypoint[head_idx, 1])
            foot_y = int(keypoint[foot_idx, 0])

            kpt = np.array([head_x, foot_y, 1])  # (x, y, 1)
            homo = M1 @ (M @ kpt.reshape((3, -1)))
            homo = np.int32(homo / homo[-1]).ravel()

            if best_mask[1] != '':
                color_bgr = hsv2bgr(COLORS[best_mask[1]][2])
                warped_kpts.append((homo, color_bgr, best_mask[1], bbox_person))
                if 0 <= head_x < frame.shape[1] and 0 <= foot_y < frame.shape[0]:
                    cv2.circle(frame, (head_x, foot_y), 2, color_bgr, 5)

        for homo, color, color_key, bbox in warped_kpts:
            iou_scores = []
            for player in self.players:
                if (player.team == color_key) and (player.previous_bb is not None) and \
                   (0 <= homo[0] < map_2d.shape[1]) and (0 <= homo[1] < map_2d.shape[0]):
                    iou_current = self.bb_intersection_over_union(bbox, player.previous_bb)
                    if iou_current >= IOU_TH:
                        iou_scores.append((iou_current, player))

            if iou_scores:
                max_iou = max(iou_scores, key=itemgetter(0))
                max_iou[1].previous_bb = bbox
                max_iou[1].positions[timestamp] = (int(homo[0]), int(homo[1]))
            else:
                for player in self.players:
                    if (player.team == color_key) and (player.previous_bb is None):
                        player.previous_bb = bbox
                        player.positions[timestamp] = (int(homo[0]), int(homo[1]))
                        break

        for player in self.players:
            if player.positions:
                if (timestamp - max(player.positions.keys())) >= 7:
                    player.positions = {}
                    player.previous_bb = None
                    player.has_ball = False

        map_2d_text = map_2d.copy()
        for p in self.players:
            if p.team != 'referee':
                try:
                    pos = p.positions[timestamp]
                    cv2.circle(map_2d, pos, 10, p.color, 7)
                    cv2.circle(map_2d, pos, 13, (0, 0, 0), 3)
                    cv2.circle(map_2d_text, pos, 25, p.color, -1)
                    cv2.circle(map_2d_text, pos, 27, (0, 0, 0), 5)
                    text_size, _ = cv2.getTextSize(str(p.ID), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                    text_origin = (pos[0] - text_size[0] // 2, pos[1] + text_size[1] // 2)
                    cv2.putText(map_2d_text, str(p.ID), text_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
                except KeyError:
                    pass

        return frame, map_2d, map_2d_text

    # ===== Alias de instancia en español =====
    def obtener_pos_jugadores(self, M, M1, frame, timestamp, map_2d):
        """Alias en español de get_players_pos(...)."""
        return self.get_players_pos(M, M1, frame, timestamp, map_2d)


# Alias de clase en español (opcional)
DetectorPies = FeetDetector
