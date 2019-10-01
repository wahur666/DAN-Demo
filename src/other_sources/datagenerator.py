from math import log2, log1p
import csv

with open('log2Data.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)

    a = {
        "vertex_num": 0,
        "avg_route_len": 0,
        "real_congestion": 0,
        "constant": 0,
        "max_delta": 0
    }
    writer.writerow(a)
    for i in [25, 50, 75, 100, 125, 150, 175, 200]:
        a = {
            "vertex_num" : i,
            "avg_route_len" : log2(log2(i)),
            "real_congestion" : 1/i,
            "constant" : 0,
            "max_delta" : 0
        }

        writer.writerow(a.values())

with open('lnData.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)

    a = {
        "vertex_num": 0,
        "avg_route_len": 0,
        "real_congestion": 0,
        "constant": 0,
        "max_delta": 0
    }
    writer.writerow(a)
    for i in [25, 50, 75, 100, 125, 150, 175, 200]:
        a = {
            "vertex_num" : i,
            "avg_route_len" : log1p(log1p(i)),
            "real_congestion" : 1/i,
            "constant" : -1,
            "max_delta" : 0
        }

        writer.writerow(a.values())
