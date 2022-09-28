import xml.etree.ElementTree as ET
import os
from pathlib import Path

xmldir_ppe = r'C:\Users\Lucas_Giam\Desktop\sp_ppe_2\VOCdevkit\VOC2012\Annotations'
xmldir_person = r'C:\Users\Lucas_Giam\Desktop\datasets_2\VOCdevkit\VOC2012\Annotations'

xmlnames_ppe = sorted(list(Path.iterdir(Path(xmldir_ppe))))
xmlnames_person = sorted(list(Path.iterdir(Path(xmldir_person))))

for i, (xmlname_ppe, xmlname_person) in enumerate(zip(xmlnames_ppe, xmlnames_person)):
    tree1 = ET.parse(xmlname_ppe)
    root1 = tree1.getroot()

    tree2 = ET.parse(xmlname_person)
    root2 = tree2.getroot()

    objects2 = root2.findall('object')

    for node in objects2:
        label_node = node.find('name')
        label_node.text = "person"

    root1.extend(objects2)

    tree1.write(xmlname_ppe)