import json

lavila_caption = json.load(
        open("/Users/maziyu/Desktop/VideoQA-Project/LLoVi-main/data/egoschema/lavila_subset.json", "r"))

lavila_caption_add_time = {}
for index, item in lavila_caption.items():
        item_caption = []
        for i, caption in enumerate(item):
                caption = str(i+1) + " " + caption
                item_caption.append(caption)
        lavila_caption_add_time[index] = item_caption

with open('lavila_subset_add_time.json', 'w') as f:
        json.dump(lavila_caption_add_time, f, indent=2)

