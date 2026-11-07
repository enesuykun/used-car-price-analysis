"""app/index.html dosyasını üretir: şablon + tahmin kodu + model katsayıları tek dosyada (sunucu gerekmez).

    python app/export_model.py     # önce modeli dışa aktar
    python app/build_app.py        # app/index.html
    python app/build_app.py --fragment yol.html   # çerçevesiz sürüm (Artifact yayını için)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
WRAP = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head}
<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}</style>
</head>
<body>
{body}
</body>
</html>
"""


def assemble() -> str:
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    model = json.dumps(json.loads((HERE / "model.json").read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))
    est = (HERE / "estimator.js").read_text(encoding="utf-8")
    assert tpl.count("/*__ESTIMATOR__*/") == 1 and tpl.count("/*__MODEL__*/null") == 1
    assert "</script" not in est and "</script" not in model
    return tpl.replace("/*__ESTIMATOR__*/", est).replace("/*__MODEL__*/null", model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="çerçevesiz (doctype/head/body olmayan) çıktı yolu")
    args = ap.parse_args()
    full = assemble()
    head, body = full.split("<!--BODY-->", 1)
    if args.fragment:
        Path(args.fragment).write_text(head + body, encoding="utf-8")
        print(f"fragment: {args.fragment} ({len(head + body) / 1024:.0f} KB)")
        return
    out = HERE / "index.html"
    out.write_text(WRAP.format(head=head.strip(), body=body.strip()), encoding="utf-8")
    print(f"yazıldı: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
