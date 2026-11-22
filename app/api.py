import base64
import json
from pathlib import Path

import webview

from . import services


class AppAPI:
    def list_datasets(self):
        return [services.dataset_info()]

    def get_dataset_info(self, name="iris"):
        info = services.dataset_info()
        if name not in ("iris", info["id"]):
            raise ValueError(f"未知数据集：{name}")
        return info

    def run_training(self, config):
        return services.run_training(config)

    def compare_variants(self, config):
        return services.compare_variants(config)

    def save_result(self, result):
        return self._save("cart-result.json", "JSON (*.json)",
                          json.dumps(result, ensure_ascii=False, indent=2).encode())

    def save_chart(self, data_url):
        prefix = "data:image/png;base64,"
        if not isinstance(data_url, str) or not data_url.startswith(prefix):
            raise ValueError("图表必须为 PNG")
        payload = base64.b64decode(data_url[len(prefix):], validate=True)
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("无效的 PNG 数据")
        return self._save("cart-tree.png", "PNG (*.png)", payload)

    def _save(self, name, file_type, payload):
        selected = webview.windows[0].create_file_dialog(
            webview.FileDialog.SAVE, save_filename=name, file_types=(file_type,)
        )
        if not selected:
            return {"saved": False, "path": None}
        path = Path(selected[0] if isinstance(selected, (tuple, list)) else selected)
        if not path.suffix:
            path = path.with_suffix(Path(name).suffix)
        path.write_bytes(payload)
        return {"saved": True, "path": str(path)}
