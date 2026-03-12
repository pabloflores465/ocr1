"""
OCR para imágenes tipo captcha con ruido (estática, líneas diagonales, letras deformadas).

Pipeline:
1. Preprocesamiento con OpenCV (eliminar ruido, líneas, normalizar)
2. Segmentación de caracteres con componentes conectados
3. Reconocimiento con EasyOCR + beamsearch (red neuronal CRNN, no LLM)
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import easyocr


def load_image(path: str) -> np.ndarray:
    """Carga imagen y valida que exista."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: no se pudo cargar la imagen '{path}'")
        sys.exit(1)
    return img


def preprocess(img: np.ndarray, debug: bool = False, debug_dir: str = "") -> np.ndarray:
    """
    Pipeline de preprocesamiento para captchas ruidosos.
    Estrategia: aislar los trazos gruesos (letras) de los delgados (líneas, puntos).
    """
    def save_debug(name: str, image: np.ndarray):
        if debug:
            cv2.imwrite(os.path.join(debug_dir, name), image)

    # 1. Escalar 3x para mejor resolución
    scale = 3
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_debug("01_gray.png", gray)

    # 3. Filtro bilateral: suaviza ruido pero preserva bordes de las letras
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    save_debug("02_filtered.png", filtered)

    # 4. Umbral de Otsu para binarizar
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_debug("03_binary.png", binary)

    # 5. Invertir: letras blancas sobre fondo negro (para morfología)
    inverted = cv2.bitwise_not(binary)
    save_debug("04_inverted.png", inverted)

    # 6. Erosión para eliminar líneas delgadas y puntos
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(inverted, kernel_erode, iterations=1)
    save_debug("05_eroded.png", eroded)

    # 7. Dilatación para restaurar grosor de letras
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)
    save_debug("06_dilated.png", dilated)

    # 8. Cerrar huecos dentro de las letras (closing)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    save_debug("07_closed.png", closed)

    # 9. Eliminar componentes conectados pequeños (restos de ruido)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    if areas:
        median_area = np.median(areas)
        min_area = max(median_area * 0.08, 50)
    else:
        min_area = 50

    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    save_debug("08_cleaned.png", cleaned)

    # 10. Invertir de vuelta: letras negras sobre fondo blanco
    result = cv2.bitwise_not(cleaned)
    save_debug("09_final.png", result)
    if debug:
        print(f"Imágenes de debug guardadas en {debug_dir}/")

    return result


def segment_characters(binary: np.ndarray, debug: bool = False, debug_dir: str = "") -> list[np.ndarray]:
    """
    Segmenta la imagen en caracteres/grupos usando componentes conectados.
    Caracteres que se tocan se pasan como grupo a EasyOCR (beamsearch los lee).
    """
    inverted = cv2.bitwise_not(binary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    h_img, w_img = binary.shape
    min_char_area = (h_img * w_img) * 0.003
    max_char_area = (h_img * w_img) * 0.5

    char_boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_char_area <= area <= max_char_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            char_boxes.append((x, y, w, h))

    char_boxes.sort(key=lambda b: b[0])

    # Fusionar componentes que se solapan horizontalmente (ej: 'i' = punto + cuerpo)
    merged = []
    for box in char_boxes:
        if merged and _boxes_overlap_x(merged[-1], box):
            merged[-1] = _merge_boxes(merged[-1], box)
        else:
            merged.append(box)

    # Extraer recortes con padding inteligente (sin invadir vecinos)
    all_crops = []
    padding = 10
    for idx, (x, y, w, h) in enumerate(merged):
        pad_left = padding
        pad_right = padding
        if idx > 0:
            prev_x, _, prev_w, _ = merged[idx - 1]
            gap_left = x - (prev_x + prev_w)
            pad_left = min(padding, max(0, gap_left // 2))
        if idx < len(merged) - 1:
            next_x, _, _, _ = merged[idx + 1]
            gap_right = next_x - (x + w)
            pad_right = min(padding, max(0, gap_right // 2))

        x0 = max(0, x - pad_left)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + pad_right)
        y1 = min(h_img, y + h + padding)
        crop = binary[y0:y1, x0:x1]
        all_crops.append(crop)

    # Redimensionar cada recorte a 128px de alto para OCR consistente
    characters = []
    for i, crop in enumerate(all_crops):
        resized = _resize_char(crop, target_h=128)
        characters.append(resized)
        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"char_{i}.png"), resized)

    return characters


def _boxes_overlap_x(box1: tuple, box2: tuple) -> bool:
    """Checa si dos bounding boxes se solapan horizontalmente (2px tolerancia)."""
    x1, _, w1, _ = box1
    x2, _, _, _ = box2
    return x2 < (x1 + w1 + 2)


def _merge_boxes(box1: tuple, box2: tuple) -> tuple:
    """Fusiona dos bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)


def _resize_char(crop: np.ndarray, target_h: int = 128) -> np.ndarray:
    """Redimensiona un recorte de carácter manteniendo proporción y centrándolo."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target_h, target_h), dtype=np.uint8) * 255

    scale = target_h / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    canvas_w = max(new_w + 20, target_h)
    canvas = np.ones((target_h + 20, canvas_w), dtype=np.uint8) * 255
    y_off = 10
    x_off = (canvas_w - new_w) // 2
    canvas[y_off:y_off + target_h, x_off:x_off + new_w] = resized

    return canvas


def ocr_image(image_path: str, debug: bool = False) -> str:
    """
    Ejecuta el pipeline completo: preprocesamiento + segmentación + OCR.
    """
    img = load_image(image_path)

    debug_dir = ""
    if debug:
        image_name = Path(image_path).stem
        debug_dir = os.path.join("debug", image_name)
        os.makedirs(debug_dir, exist_ok=True)

    clean = preprocess(img, debug=debug, debug_dir=debug_dir)
    characters = segment_characters(clean, debug=debug, debug_dir=debug_dir)

    if not characters:
        return ""

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    allowlist = "abcdefghijklmnopqrstuvwxyz0123456789"

    detected = []
    for i, char_img in enumerate(characters):
        # Usar beamsearch para mejor decodificación de multi-caracteres
        results = reader.readtext(
            char_img,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            beamWidth=10,
            text_threshold=0.1,
            low_text=0.2,
            allowlist=allowlist,
        )

        if not results:
            # Fallback: recognize() salta el detector de texto
            h, w = char_img.shape[:2]
            if h > 2 and w > 2:
                results = reader.recognize(
                    char_img,
                    horizontal_list=[[0, w, 0, h]],
                    free_list=[],
                    decoder="beamsearch",
                    beamWidth=10,
                    allowlist=allowlist,
                )

        if results:
            # Concatenar todos los textos detectados (ordenados por x)
            results.sort(key=lambda r: r[0][0][0] if isinstance(r[0][0], (list, tuple)) else r[0][0])
            text = "".join(r[1].strip().lower() for r in results if r[1].strip())
            avg_conf = np.mean([r[2] for r in results]) if results else 0
            if text:
                detected.append(text)
                if debug:
                    print(f"  char_{i}: '{text}' (confianza: {avg_conf:.2f})")
            elif debug:
                print(f"  char_{i}: <no detectado>")
        elif debug:
            print(f"  char_{i}: <no detectado>")

    return "".join(detected)


def main():
    if len(sys.argv) < 2:
        print("Uso: python ocr_captcha.py <imagen> [--debug]")
        print("Ejemplo: python ocr_captcha.py captcha.png --debug")
        sys.exit(1)

    image_path = sys.argv[1]
    debug = "--debug" in sys.argv

    print(f"Procesando: {image_path}")
    result = ocr_image(image_path, debug=debug)
    print(f"Texto detectado: {result}")


if __name__ == "__main__":
    main()
