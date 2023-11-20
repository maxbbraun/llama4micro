import yaml

YAML_FILE = 'yolov5/data/coco.yaml'
TXT_FILE = 'models/yolov5/coco_labels.txt'

# Open the YAML file containing the COCO labels.
with open(YAML_FILE) as f:
    coco128_yaml = yaml.safe_load(f)

# Extract the labels from the data structure.
labels = coco128_yaml['names']

print(f'Opened {YAML_FILE} containing {len(labels)} labels.')

# Save one label per line to a text file.
with open(TXT_FILE, 'w') as f:
    for i in range(len(labels)):
        f.write(f'{labels[i]}\n')

print(f'Saved {len(labels)} labels to {TXT_FILE}.')
