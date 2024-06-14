import numpy as np

desc = np.array([["--------"], ["-x--x---"], ["x----xx-"],["L--x----"],["--xxx--S"], ["-x--x---"], ["--------"], ["-x----x-"]])
desc = np.array([list(row[0]) for row in desc])

print(desc[0,0])