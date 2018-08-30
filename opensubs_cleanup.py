import os
import xml.etree.ElementTree as ElementTree
from shutil import copyfile

src_base_path = "corpora/opensubs/xml/"
tree = ElementTree.parse(src_base_path + "en-ja.xml")
root = tree.getroot()

src_raw_base_path = src_base_path + "raw/"
src_tokenized_base_path = src_base_path + "tokenized/"

dst_base_path = "corpora/clean/opensubs/xml/"
dst_raw_base_path = dst_base_path + "raw/"
dst_tokenized_base_path = dst_base_path + "tokenized/"

for link_group in root.findall("linkGrp"):
    specific_path = link_group.attrib["fromDoc"]
    src_raw_path = src_raw_base_path + specific_path
    src_tokenized_path = src_tokenized_base_path + specific_path
    dst_raw_path = dst_raw_base_path + specific_path
    dst_tokenized_path = dst_tokenized_base_path + specific_path
    os.makedirs(os.path.dirname(dst_raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(dst_tokenized_path), exist_ok=True)
    copyfile(src_raw_path, dst_raw_path)
    copyfile(src_tokenized_path, dst_tokenized_path)
