import sys

import webview

from app.api import AppAPI
from app.resources import FRONTEND


def main():
    gui = {"linux": "qt", "win32": "edgechromium"}.get(sys.platform)
    if gui is None:
        raise SystemExit("CARTLab 支持 Linux (Qt) 与 Windows (WebView2)")
    webview.create_window(
        "CARTLab · 决策树实验室",
        str(FRONTEND),
        js_api=AppAPI(),
        width=1440,
        height=920,
        min_size=(900, 640),
    )
    webview.start(gui=gui)


if __name__ == "__main__":
    main()
