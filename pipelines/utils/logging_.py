import os


def primary_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)
