path = "DATASET/coco/annotations/lvis_v1_minival.json"
import json
with open(path) as f:
    all = json.load(f)

for i in all["images"]:
    i["file_name"] = "/".join(i["coco_url"].split("/")[-2:])

with open("DATASET/coco/annotations/lvis_v1_minival_inserted_image_name.json", "w") as f:
    json.dump(all, f)