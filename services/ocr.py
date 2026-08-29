import logging
import easyocr
import numpy as np

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, languages=['en']):
        logger.info(f"Loading EasyOCR model for languages: {languages}...")
        try:
            self.reader = easyocr.Reader(languages, gpu=True)
            logger.info("EasyOCR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR model: {e}")
            self.reader = None

    def detect_text(self, frame: np.ndarray, conf_threshold: float = 0.5):
        if self.reader is None:
            return {"texts": []}

        try:
            results = self.reader.readtext(frame)
        except Exception as e:
            logger.error(f"EasyOCR readtext error: {e}")
            return {"texts": []}

        texts = []
        for (bbox, text, prob) in results:
            if prob >= conf_threshold:
                serializable_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
                texts.append({
                    "text": text,
                    "confidence": round(float(prob), 3),
                    "bbox": serializable_bbox
                })

        return {"texts": texts}
