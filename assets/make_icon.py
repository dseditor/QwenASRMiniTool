"""make_icon.py — 由 codex 生成的白底貓耳毛絨小耳機產出程式圖示。

去背策略：主體是「白色毛絨」疊在「白色背景」上，全域白鍵會吃掉毛絨本體，
故改用**從四角洪水填充（flood fill）**：只移除「與邊界相連」的外圍背景白，
主體被自身邊緣／陰影包住、不與邊界相連 → 內部白毛絨完整保留。

產出：
  icon.png  —  256px 透明去背（網頁 favicon / 視窗圖示用）
  icon.ico  —  多尺寸 (16/32/48/64/128/256) 透明圖示（PyInstaller --icon / 視窗）
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).resolve().parent
SRC = ASSETS / "icon_src.png"
SENTINEL = (255, 0, 255)          # 洪水填充用的標記色（圖中不存在）
THRESH = 36                       # 與起點顏色差異容忍度（白背景連通區）


def cutout(src: Path) -> Image.Image:
    """從四角洪水填充移除外圍背景白，回傳 RGBA（背景透明）。"""
    rgb = Image.open(src).convert("RGB")
    w, h = rgb.size
    # 在 RGB 副本上，從四角把連通的近白背景填成 SENTINEL
    flood = rgb.copy()
    for xy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
               (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]:
        ImageDraw.floodfill(flood, xy, SENTINEL, thresh=THRESH)

    # alpha：SENTINEL 像素 → 0（透明），其餘 → 255
    px = flood.load()
    alpha = Image.new("L", (w, h), 255)
    ap = alpha.load()
    for y in range(h):
        for x in range(w):
            if px[x, y] == SENTINEL:
                ap[x, y] = 0
    # 邊緣羽化一點，去除鋸齒（毛絨邊本就柔和）
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.6))

    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def square_trim(img: Image.Image, pad_ratio: float = 0.06) -> Image.Image:
    """裁到非透明內容的外接框，置中放進正方畫布並留邊。"""
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    w, h = img.size
    side = max(w, h)
    pad = int(side * pad_ratio)
    canvas = Image.new("RGBA", (side + 2 * pad, side + 2 * pad), (0, 0, 0, 0))
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2), img)
    return canvas


def main():
    if not SRC.exists():
        raise SystemExit(f"找不到來源圖：{SRC}")
    cut = cutout(SRC)
    cut = square_trim(cut)

    png = cut.resize((256, 256), Image.LANCZOS)
    png.save(ASSETS / "icon.png")
    print(f"saved {ASSETS / 'icon.png'}")

    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    cut.save(ASSETS / "icon.ico", format="ICO", sizes=sizes)
    print(f"saved {ASSETS / 'icon.ico'} sizes={sizes}")


if __name__ == "__main__":
    main()
