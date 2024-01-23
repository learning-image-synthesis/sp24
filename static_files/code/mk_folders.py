"""
    This file shows how to load and use the dataset
"""

import argparse
import json
from pathlib import Path
import csv
import subprocess

parser = argparse.ArgumentParser(description="Make a project folder for each student")
parser.add_argument("--root", type=str, default="/afs/andrew.cmu.edu/course/16/726-sp24/",
                    help="Root Course Web Project Volume")
parser.add_argument("--roster_file", type=str, default="piazza-16726_roster.csv",
                    help="Roster file name.")

def main(args):
    root = Path(args.root)
    roster_file = root / args.roster_file
    www_root = root / 'www' / 'projects'
    assert roster_file.exists(), "Roster file does not exist. Please export from piazza"
    if not www_root.exists():
        www_root.mkdir(exist_ok=True)
    with open(roster_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['name'], row['email'], row['role'])
            andrew_id = row['email'][:row['email'].index('@')]
            print(andrew_id)
            student_folder = www_root / andrew_id
            subprocess.run(
               ['mkdir', '-p', str(student_folder)],
               text=True,
               stdout=subprocess.PIPE,
               check=True
            )
            subprocess.run(
               ['fs', 'sa', str(student_folder), andrew_id, 'write'],
               text=True,
               stdout=subprocess.PIPE,
               check=True
            )
            html_str = f"<!doctype html><html lang=en><head><meta charset=utf-8><title>{andrew_id}</title></head>\n" + \
                       f"<body><p>{row['name']}'s home page for 16-726</p></body></html>" 
            with open(str(student_folder / "index.html"), "w+") as f:
                f.write(html_str)

            student_folder_0 = www_root / andrew_id / "proj0"
            subprocess.run(
               ['mkdir', '-p', str(student_folder_0)],
               text=True,
               stdout=subprocess.PIPE,
               check=True
            )

            html_str = f"<!doctype html><html lang=en><head><meta charset=utf-8><title>{andrew_id}'s project 0</title></head>\n" + \
                       f"<body><p>{row['name']}'s home page for 16-726 project 0</p></body></html>" 
            with open(str(student_folder_0 / "index.html"), "w+") as f:
                f.write(html_str)
        

if __name__ == "__main__":
    main(parser.parse_args())
