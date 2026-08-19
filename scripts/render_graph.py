from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "graph.png"


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


canvas = Image.new("RGB", (1800, 1200), "#F7F5F0")
draw = ImageDraw.Draw(canvas)
draw.text((90, 55), "CityScope · LangGraph topology", fill="#172A2B", font=font(44, True))
draw.text((90, 112), "Conditional knowledge routing, parallel data fan-out, and memory-aware refresh", fill="#5D6969", font=font(22))

boxes = {
    "START": (70, 250, 240, 330),
    "analyze_request": (310, 230, 630, 350),
    "check_knowledge": (700, 125, 1010, 245),
    "refresh_weather": (700, 405, 1010, 525),
    "retrieve_vector": (1080, 65, 1370, 165),
    "search_web": (1080, 255, 1370, 355),
    "prepare_summary": (1430, 165, 1750, 265),
    "fetch_weather": (1100, 535, 1400, 635),
    "fetch_images": (1460, 535, 1760, 635),
    "finalize_refresh": (665, 810, 995, 910),
    "finalize": (1285, 785, 1585, 885),
    "END": (1515, 1000, 1685, 1080),
}


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def arrow(a, b, label=""):
    start, end = center(boxes[a]), center(boxes[b])
    draw.line((start, end), fill="#557170", width=5)
    angle_x, angle_y = end[0] - start[0], end[1] - start[1]
    length = max((angle_x**2 + angle_y**2) ** 0.5, 1)
    ux, uy = angle_x / length, angle_y / length
    tip = (end[0] - ux * 65, end[1] - uy * 45)
    left = (tip[0] - ux * 18 - uy * 12, tip[1] - uy * 18 + ux * 12)
    right = (tip[0] - ux * 18 + uy * 12, tip[1] - uy * 18 - ux * 12)
    draw.polygon([tip, left, right], fill="#557170")
    if label:
        mx, my = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
        draw.rounded_rectangle((mx - 78, my - 20, mx + 78, my + 20), 12, fill="#F7F5F0")
        draw.text((mx, my), label, anchor="mm", fill="#A0472E", font=font(18, True))


for a, b, label in [
    ("START", "analyze_request", ""),
    ("analyze_request", "check_knowledge", "full request"),
    ("analyze_request", "refresh_weather", "follow-up"),
    ("check_knowledge", "retrieve_vector", "known"),
    ("check_knowledge", "search_web", "unknown"),
    ("retrieve_vector", "prepare_summary", ""),
    ("search_web", "prepare_summary", ""),
    ("prepare_summary", "fetch_weather", "parallel"),
    ("prepare_summary", "fetch_images", "parallel"),
    ("fetch_weather", "finalize", "join"),
    ("fetch_images", "finalize", "join"),
    ("refresh_weather", "finalize_refresh", "keeps summary"),
    ("finalize_refresh", "END", ""),
    ("finalize", "END", ""),
]:
    arrow(a, b, label)

for name, box in boxes.items():
    fill = "#172A2B" if name in {"START", "END"} else "#FFFFFF"
    outline = "#172A2B" if name in {"START", "END"} else "#D6CEC1"
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=4)
    color = "white" if name in {"START", "END"} else "#172A2B"
    label = name.replace("_", "\n")
    draw.multiline_text(center(box), label, anchor="mm", align="center", fill=color, font=font(24, True), spacing=5)

draw.rounded_rectangle((90, 1120, 1710, 1180), radius=20, fill="#EAE4D9")
draw.text((120, 1150), "MemorySaver checkpoints state by thread_id · mock mode is offline · live mode uses public APIs", anchor="lm", fill="#4A5555", font=font(20))
canvas.save(OUT)
print(OUT)
